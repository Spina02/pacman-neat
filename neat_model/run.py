import os
import neat
import sys
import pickle
import pygame
import time
import argparse
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

pacman_game_path = os.path.join(project_root, "neat", "game", "pacman")
if pacman_game_path not in sys.path:
    sys.path.insert(0, pacman_game_path)

try:
    from game.game_env import PacmanEnvironment
except ModuleNotFoundError:
    print("Error: Could not find PacmanEnvironment.")
    sys.exit(1)

EASY_GEN = 300
DEFAULT_CONFIG_PATH = os.path.join(project_root, 'config')
DEFAULT_CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints')

class RunBestGenome:
    def __init__(self, config_path, checkpoint_path, debug=False):
        self.debug = debug
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"NEAT configuration not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"NEAT checkpoint not found: {checkpoint_path}")

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.best_genome = None
        self.network = None
        self.env = None
        try:
            self.current_gen = int(self.checkpoint_path.split('-')[-1].split('.')[0])
        except ValueError:
            self.current_gen = 0  # Fallback if the filename does not follow the usual pattern

        print(f"Loading NEAT config from: {self.config_path}")
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        self.population = None
        print(f"Restoring population from checkpoint: {self.checkpoint_path}")
        if self.checkpoint_path.endswith('.pkl'):
            with open(self.checkpoint_path, 'rb') as f:
                winner_genome = pickle.load(f)

            # Then rebuild the NEAT config
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                self.config_path
            )
            # Finally recreate the network
            self.network = neat.nn.FeedForwardNetwork.create(winner_genome, config)
            self.best_genome = winner_genome  # Use the loaded genome as the best_genome
        else:
            self.population = neat.Checkpointer.restore_checkpoint(self.checkpoint_path)
            self.best_genome = self.population.best_genome

        if self.best_genome is None:
            print("Warning: best_genome is None. Finding best in population.")
            genomes = list(self.population.population.values())
            if not genomes:
                raise ValueError("No genomes in checkpoint!")
            genomes.sort(key=lambda g: g.fitness if g.fitness else -float('inf'), reverse=True)
            self.best_genome = genomes[0]

        print(f"Best genome found: ID={self.best_genome.key}, Fitness={self.best_genome.fitness:.2f}")
        print("Creating neural network from the best genome...")
        self.network = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)

        print("Initializing Pacman environment for visualization...")
        self.env = PacmanEnvironment(render=True)

    def run(self, max_steps=5000):
        if not self.network or not self.env:
            print("Error: Network or Environment not initialized.")
            return

        print("\n--- Starting Pacman Game Run ---")
        try:
            self.env.current_gen = self.current_gen
            obs = self.env.reset()
            if self.current_gen < EASY_GEN:
                self.env.game_state.no_ghosts = True
                
            done = False
            total_reward = 0
            step_count = 0

            if self.debug: print("[action, pacman_pos, reward] for each step:")
            running = True
            while running and not done and step_count < max_steps:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("QUIT event received, stopping run.")
                        running = False
                        break
                if not running:
                    break

                obs_array = np.array(obs)
                outputs = self.network.activate(obs_array)
                action = np.argmax(outputs)

                obs, reward, done, info = self.env.step(action)
                
                total_reward += reward
                
                if self.debug: print(f"[{action}, [{self.env.pacman_pos[0]:.3f},{self.env.pacman_pos[1]:.3f}] , {reward:.3f}", end = "],")
                #print(total_reward)
                
                step_count += 1

                time.sleep(1 / 120)

            print("\n--- Game Run Finished ---")
            print(f"Reason: {'Completed level / Died' if done else 'Window closed' if not running else 'Max steps reached'}")
            print(f"Total steps: {step_count}")
            print(f"Final Score (from game state): {self.env.game_state.points}")
            print(f"Total Accumulated Reward: {total_reward:.2f}")

        except Exception as e:
            print(f"\nError during the run: {e}")
        finally:
            if self.env:
                self.env.close()
            print("Environment closed.")

if __name__ == "__main__":
    
    DEBUG = False
    
    parser = argparse.ArgumentParser(description="Run best Pacman genome from NEAT checkpoint.")
    parser.add_argument("checkpoint_file", type=str,
                        help="Path to the NEAT checkpoint file.")
    parser.add_argument("--config", type=str,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to the NEAT config file.")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("NEAT Pacman - Best Genome Run")

    try:
        runner = RunBestGenome(config_path=args.config, checkpoint_path=args.checkpoint_file, debug=DEBUG)
        runner.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        pygame.quit()
        sys.exit(1)

    pygame.quit()
    print("Pygame quit.")
