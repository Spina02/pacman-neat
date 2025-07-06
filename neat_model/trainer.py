import os
import neat
import sys
import multiprocessing as mp
import numpy as np
import pickle
import signal
import sys
import pygame
import gc
from .neat_utils import BestGenomeSaver
import random

seed = 42
random.seed(seed)
np.random.seed(seed)

_pool = None
_worker_env = None

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'config'))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'checkpoints'))
BEST_GENOME_DIR = os.path.join(CHECKPOINT_DIR, 'best_genomes')

def init_worker(render_mode, obs_mode):
    """
    Initialization function for each process in the pool.
    Creates a single environment instance for this process.
    """
    global _worker_env
    from game import game_env as game
    _worker_env = game.PacmanEnvironment(render=render_mode, observation_mode=obs_mode)
    _worker_env._is_worker = True # Flag to handle pygame shutdown

def cleanup_resources():
    """Cleanup function to be called at exit"""
    global _pool
    
    print("Cleaning up resources...")
    
    # Clean up pool
    if _pool:
        try:
            _pool.terminate()
            _pool.join()
        except:
            pass
    
    # Ensure pygame is quit
    if pygame.get_init():
        try:
            pygame.quit()
        except:
            pass
    
    # Force garbage collection
    _pool = None
    gc.collect()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if (project_root not in sys.path):
    sys.path.insert(0, project_root)

try:
    from game import game_env as game
except ImportError as e:
    print(f"Error importing game_env: {e}")
    print("Please check the path and project structure.")
    sys.exit(1)

def _handle_sigint(pool):
    """Signal handler for Ctrl+C that cleans up the multiprocessing pool."""
    def handler(signum, frame):
        """Signal handler for both SIGINT and SIGTERM"""
        print(f"Signal {signum} caught; cleaning up...")
        cleanup_resources()
        sys.exit(0)
    return handler

def evaluate_genome(args):
    """
    Evaluation function that now reuses the environment from its worker.
    """
    global _worker_env
    current_gen, genome_id, genome, config = args

    # Ensure the environment has been initialized
    if _worker_env is None:
        print(f"FATAL: Worker {os.getpid()} environment not initialized!")
        return (genome_id, -float('inf')) # Return a terrible fitness
    
    env = _worker_env # Use the worker's environment

    try:
        env.current_gen = current_gen
        env.debug = 0
        state = env.reset()
            
        # Create the appropriate NEAT network (FF or Recurrent)
        if config.genome_config.feed_forward:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            
        done = False
        total_reward = 0
        genome.fitness = 0
        while not done:
            outputs = net.activate(state)
            action = np.argmax(outputs)
            state, reward, done = env.step(action)
            total_reward += reward
        EXP_BONUS = 4
        reward = env.tot_visited * EXP_BONUS
        total_reward += reward
        if env.debug >= 3: print(f"Exploration bonus: {reward}")
    except Exception as e:
        print(f"Error in evaluating genome {genome_id} in worker {os.getpid()}: {e}")
        import traceback
        traceback.print_exc()
        total_reward = -float('inf')
    
    # We don't close the environment here, it will be reused.
    return (genome_id, total_reward)

class Trainer:
    def __init__(self, neat_config_file=CONFIG_PATH, gen=2000, cores=15, resume_from=None, render=False, observation_mode='minimap'):
        self.gen = gen
        self.cores = cores
        self.render = render
        self.observation_mode = observation_mode # Store observation mode

        print(f"Initializing Trainer with Observation Mode: {self.observation_mode}")

        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_file
        )

        expected_inputs = 90 if self.observation_mode == 'minimap' else 26
        if self.neat_config.genome_config.num_inputs != expected_inputs:
            print(f"FATAL ERROR: NEAT config 'num_inputs' ({self.neat_config.genome_config.num_inputs}) "
                  f"does not match expected inputs for observation_mode='{self.observation_mode}' ({expected_inputs}).")
            print("Please update your NEAT config file ('config') and restart.")
            sys.exit(1)

        if resume_from:
            print(f"Attempting to restore checkpoint: {resume_from}")
            try:
                self.population = neat.Checkpointer.restore_checkpoint(resume_from)
                print(f"Successfully resumed from generation {self.population.generation}")
            except Exception as e:
                print(f"Error restoring checkpoint {resume_from}: {e}")
                print("Starting a new population.")
                self.population = neat.Population(self.neat_config)
        else:
            print("No checkpoint provided or restoration failed. Starting a new population.")
            self.population = neat.Population(self.neat_config)

        # Store the current generation number from the population object
        self.current_gen = self.population.generation

        # Add reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        if not os.path.exists(CHECKPOINT_DIR):
            print(f"Creating checkpoint directory: {CHECKPOINT_DIR}")
            os.makedirs(CHECKPOINT_DIR)
        prefix = os.path.join(CHECKPOINT_DIR, f'checkpoint-{self.observation_mode}-') # Include mode in checkpoint name
        self.population.add_reporter(
            neat.Checkpointer(generation_interval=5, filename_prefix=prefix)
        )
        self.population.add_reporter(neat.StatisticsReporter())
        self.population.add_reporter(BestGenomeSaver(save_path=BEST_GENOME_DIR, filename_prefix=f'best_{self.observation_mode}', generation = self.current_gen))

        # Initialize the multiprocessing pool if cores > 1
        global _pool
        self.pool = None
        if self.cores > 1:
            # Use 'spawn' context for better compatibility across platforms, especially with Pygame
            mp_context = mp.get_context('spawn')
            self.pool = mp_context.Pool(
                processes=self.cores,
                initializer=init_worker,
                initargs=(self.render, self.observation_mode) # Arguments for initializer
            )
            _pool = self.pool # Assign to global for cleanup handler
            print(f"Multiprocessing pool initialized with {self.cores} cores (context: spawn).")
        else:
            print("Running evaluation sequentially (cores=1).")

        signal.signal(signal.SIGINT, _handle_sigint(self.pool))


    def _eval_genomes(self, genomes, config):
        """ Evaluates multiple genomes, potentially in parallel. """
        # Prepare arguments for each genome evaluation
        # Pass the current generation number from the population object
        args_list = [
            (self.population.generation, genome_id, genome, config)
            for genome_id, genome in genomes
        ]

        if self.pool:
            # Parallel evaluation using the pool
            results = self.pool.map(evaluate_genome, args_list)
        else:
            # Sequential evaluation needs its own environment
            global _worker_env
            if _worker_env is None:
                from game import game_env as game
                _worker_env = game.PacmanEnvironment(self.render, self.observation_mode)
            results = [evaluate_genome(args) for args in args_list]

        # Update genome fitness based on results
        genome_dict = {genome_id: genome for genome_id, genome in genomes}
        for genome_id, fitness in results:
             if genome_id in genome_dict:
                 # Ensure fitness is assigned correctly from the return value
                 genome_dict[genome_id].fitness = fitness
             else:
                 print(f"Warning: Genome ID {genome_id} from results not found in current generation.")


    def run_training(self):
        """ Starts the NEAT training process. """
        print(f"\n--- Starting NEAT Training ---")
        print(f"Config File: {CONFIG_PATH}")
        print(f"Observation Mode: {self.observation_mode}")
        print(f"Max Generations: {self.gen}")
        print(f"Cores for Evaluation: {self.cores}")
        print(f"Starting from Generation: {self.population.generation}")
        print("-" * 30)

        try:
            # Run NEAT's evolutionary process
            winner = self.population.run(self._eval_genomes, self.gen) # Pass target number of generations

            # Training finished, save the winner
            print("\n--- Training Finished ---")
            print(f"Best genome found: {winner}")

            # Save the winning genome
            winner_path = os.path.join(CHECKPOINT_DIR, f'winner-{self.observation_mode}.pkl')
            with open(winner_path, 'wb') as f:
                pickle.dump(winner, f)
            print(f"Winner genome saved to {winner_path}")

            # save the entire final population state
            final_pop_path = os.path.join(CHECKPOINT_DIR, f'final_population-{self.observation_mode}.pkl')
            with open(final_pop_path, 'wb') as f:
                pickle.dump(self.population, f)
            print(f"Final population saved to {final_pop_path}")

        except Exception as e:
            print(f"\n--- ERROR DURING TRAINING ---")
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up the multiprocessing pool
            if self.pool:
                print("Closing multiprocessing pool...")
                self.pool.close()
                self.pool.join()
                print("Pool closed.")

def run(restore_checkpoint,  observation_mode,config_file, generations, render, cores):

    # Create Trainer instance with the observation mode
    trainer = Trainer(
        neat_config_file=config_file,
        gen=generations,
        cores=cores,
        resume_from=restore_checkpoint,
        render=render,
        observation_mode=observation_mode
    )

    trainer.run_training()