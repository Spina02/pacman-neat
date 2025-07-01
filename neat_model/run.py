# --- START OF FILE run.py ---

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

DEFAULT_CONFIG_PATH = os.path.join(project_root, 'config')
DEFAULT_CHECKPOINT_DIR = os.path.join(project_root, 'checkpoints')
DEFAULT_BEST_GENOME_DIR = os.path.join(DEFAULT_CHECKPOINT_DIR, 'best_genomes')

class RunGenome:
    def __init__(self, config_path, genome_to_load, observation_mode='minimap', is_best_overall=False, debug=0):
        self.debug = debug
        self.observation_mode = observation_mode
        self.is_best_overall = is_best_overall

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"NEAT configuration not found: {config_path}")
        if not os.path.exists(genome_to_load):
            if not is_best_overall or not genome_to_load.endswith(f'best_{observation_mode}_latest.pkl'):
                 raise FileNotFoundError(f"Genome file or checkpoint not found: {genome_to_load}")

        self.config_path = config_path
        self.load_path = genome_to_load
        self.best_genome = None
        self.network = None
        self.env = None

        print(f"Loading NEAT config from: {self.config_path}")
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )

        print(f"Attempting to load genome/checkpoint from: {self.load_path}")
        if self.load_path.endswith('.pkl'):
            try:
                with open(self.load_path, 'rb') as f:
                    self.best_genome = pickle.load(f)
                print(f"Successfully loaded genome PKL: ID={self.best_genome.key}, Fitness={getattr(self.best_genome, 'fitness', 'N/A'):.2f}")
                try:
                    if '_gen' in self.load_path:
                        gen_str = self.load_path.split('_gen')[-1].split('_')[0]
                        self.current_gen = int(gen_str)
                    else:
                        print("Warning: Unable to extract generation from filename. Setting to 9999.")
                        
                        self.current_gen = 9999
                except:
                    print("Warning: Unable to extract generation from filename. Setting to 9999.")
                    self.current_gen = 9999
            except FileNotFoundError:
                 if self.is_best_overall:
                    print("Warning: Latest best genome file not found. No simulation possible for '--best'.")
                    raise FileNotFoundError("Latest best genome file not found. Run training first.")
                 else:
                    print(f"Error: Specified PKL file not found: {self.load_path}")
                    raise
            except Exception as e:
                print(f"Error loading genome PKL: {e}")
                raise

        else: # NEAT checkpoint
             print(f"Restoring population from checkpoint: {self.load_path}")
             try:
                population = neat.Checkpointer.restore_checkpoint(self.load_path)
                if not self.is_best_overall:
                    self.best_genome = population.best_genome
                    if self.best_genome is None:
                        print("Warning: best_genome is None in checkpoint. Finding best in population.")
                        genomes = list(population.population.values())
                        if not genomes: raise ValueError("No genomes in checkpoint!")
                        genomes.sort(key=lambda g: g.fitness if g.fitness is not None else -float('inf'), reverse=True)
                        self.best_genome = genomes[0]
                    print(f"Using best genome from checkpoint: ID={self.best_genome.key}, Fitness={getattr(self.best_genome, 'fitness', 'N/A'):.2f}")
                    # Imposta la generazione dal checkpoint
                    self.current_gen = population.generation
                else:
                    # Questo caso non dovrebbe accadere se il caricamento PKL funziona
                    print("Error: --best flag was used but couldn't load PKL, and checkpoint loading is ambiguous for best overall.")
                    raise ValueError("Cannot determine best overall genome from checkpoint when --best is specified.")

             except Exception as e:
                print(f"Error restoring checkpoint: {e}")
                raise

        if self.best_genome is None:
            raise ValueError("Could not load or find a genome to run.")

        print(f"Genome to run: ID={self.best_genome.key}, Fitness={getattr(self.best_genome, 'fitness', 'N/A'):.2f}")
        print("Creating neural network from the genome...")
        self.network = neat.nn.FeedForwardNetwork.create(self.best_genome, self.config)

        print("Initializing Pacman environment for visualization...")
        self.env = PacmanEnvironment(render=True, observation_mode=self.observation_mode)
        
        print(f"Setting environment generation context to: {self.current_gen}")
        self.env.current_gen = self.current_gen

    def run(self, max_steps=None):
        if not self.network or not self.env:
            print("Error: Network or Environment not initialized.")
            return

        print("\n--- Starting Pacman Game Run ---")
        run_max_steps = self.env.MAX_EPISODE_STEPS if max_steps is None else max_steps
        print(f"Using MAX_EPISODE_STEPS: {run_max_steps}")

        try:
            obs = self.env.reset()
            self.env.debug = self.debug
            
            done = False
            total_reward = 0
            step_count = 0

            #if self.debug >= 3: print("[Step, Action, Reward, CumulativeReward]")
            running = True
            while running and not done and step_count < run_max_steps:
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

                # if self.debug >= 3:
                #      print(f"[{step_count}, {action}, {reward:.3f}, {total_reward:.3f}]")

                step_count += 1

                # Riduci il time.sleep per una visualizzazione più fluida
                time.sleep(1 / 120) # 120 FPS target

            EXP_BONUS = 4 # Riconferma il valore
            exploration_reward = self.env.tot_visited * EXP_BONUS
            total_reward += exploration_reward

            print("\n--- Game Run Finished ---")
            print(f"Exploration bonus added: {exploration_reward}")
            reason = "Unknown"
            if done:
                if self.env.game_state.level_complete:
                    reason = "Completed level" 
                elif self.env.max_reached:
                    reason = "Max steps reached" 
                else:
                    reason = "Pacman died"
            elif not running:
                 reason = "Window closed"
            print(f"Reason: {reason}")
            print(f"Total steps: {step_count}")
            print(f"Final Score (from game state): {self.env.game_state.points}")
            print(f"Total Accumulated Reward (incl. exploration): {total_reward:.2f}")

        except Exception as e:
            print(f"\nError during the run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.env:
                self.env.close()
            print("Environment closed.")
            
def run(config_file, checkpoint_file, observation_mode, best, max_steps=None, debug=0):
    """
    Initializes and runs a single genome against the Pacman environment.

    Args:
        config_file (str): Path to the NEAT configuration file.
        checkpoint_file (str): Path to the checkpoint or pickled genome to load.
        observation_mode (str): The observation mode to use ('simple' or 'minimap').
        best (bool): If True, loads the latest best genome for the given observation mode.
        max_steps (int, optional): Maximum number of steps to run the simulation. Defaults to None.
        debug (int, optional): Debug level for verbose output. Defaults to 0.
    """
    pygame.init()
    pygame.display.set_caption("NEAT Pacman - Genome Run")

    genome_to_load_path = checkpoint_file
    is_best_overall = best

    if best:
        if not os.path.isdir(DEFAULT_BEST_GENOME_DIR):
            print(f"Error: Best genome directory not found at '{DEFAULT_BEST_GENOME_DIR}'.")
            print("Please run training first to generate best genomes.")
            sys.exit(1)

        bests = [f for f in os.listdir(DEFAULT_BEST_GENOME_DIR)
                 if f.startswith(f"best_{observation_mode}_gen") and f.endswith(".pkl")]
        if not bests:
            print(f"No matching best genome found for mode '{observation_mode}' in '{DEFAULT_BEST_GENOME_DIR}'.")
            sys.exit(1)

        bests.sort(
            key=lambda f: os.path.getmtime(os.path.join(DEFAULT_BEST_GENOME_DIR, f)),
            reverse=True
        )
        genome_to_load_path = os.path.join(DEFAULT_BEST_GENOME_DIR, bests[0])
        print(f"Using latest best genome: {genome_to_load_path}")
    
    elif not checkpoint_file:
        print("Error: You must specify a checkpoint file to load (e.g., --restore_checkpoint) if not using --best.")
        sys.exit(1)

    try:
        runner = RunGenome(config_path=config_file,
                           genome_to_load=genome_to_load_path,
                           observation_mode=observation_mode,
                           is_best_overall=is_best_overall,
                           debug=debug)
        runner.run(max_steps=max_steps)
    except (FileNotFoundError, TypeError) as e:
        print(f"Error: Could not find or load file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
    finally:
        if pygame.get_init():
            pygame.quit()
        print("Pygame quit and resources released.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a Pacman genome from NEAT.")
    parser.add_argument("--load_file", type=str, default=None,
                        help="Optional: Path to the NEAT checkpoint file OR a specific .pkl genome file.")
    parser.add_argument("--best", action='store_true',
                        help="Load the best genome found overall (looks for 'best_MODE_latest.pkl'). Overrides load_file if found.")
    parser.add_argument("--config", type=str,
                        default=DEFAULT_CONFIG_PATH,
                        help="Path to the NEAT config file.")
    parser.add_argument('--observation_mode', type=str, default='minimap',
                         choices=['simple', 'minimap'], help='Observation mode used during training.')
    parser.add_argument('--max_steps', type=int, default=None,
                         help="Override the maximum steps for the run (default: use environment's MAX_EPISODE_STEPS).")
    parser.add_argument('--debug', type=int, default=0,
                        help="Debug level for verbose output (0-3).")

    args = parser.parse_args()

    run(
        config_file=args.config,
        checkpoint_file=args.load_file,
        observation_mode=args.observation_mode,
        best=args.best,
        max_steps=args.max_steps,
        debug=args.debug
    )