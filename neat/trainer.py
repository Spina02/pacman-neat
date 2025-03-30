import os
import neat
import sys
import multiprocessing as mp
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import game.game_env as game

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'config'))
CHECKPOINT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'checkpoints'))

class Trainer:
    def __init__(self, neat_config_file = CONFIG_PATH, gen = 1000, cores = 14, resume_from=None, render=False):
        self.gen = gen
        self.cores = cores
        self.lock = mp.Lock() # This is useful for synchronizing access to shared resources
        self.render = render
            
        self.neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_file
        )
        
        if resume_from:
            self.population = neat.Checkpointer.restore_checkpoint(resume_from)
            self.gen = self.population.generation
            print(f"Resuming from generation {self.gen}")
        else:
            self.population = neat.Population(self.neat_config)
            # Solo se NON stai riprendendo da checkpoint
        self.population.add_reporter(neat.StdOutReporter(True))
        prefix = os.path.join(CHECKPOINT_DIR, 'checkpoint-')
        self.population.add_reporter(neat.Checkpointer(
            generation_interval=5,
            filename_prefix=prefix
        ))
        self.population.add_reporter(neat.StatisticsReporter())
        
    def _fitness_func(self, genome, config, output):
        try:
            env = game.PacmanEnvironment(self.render)
            state = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            done = False
            total_reward = 0
            genome.fitness = 0
            
            while not done:
                outputs = net.activate(state)
                action = np.argmax(outputs)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                genome.fitness += reward
                
            # if total_reward >= 6480:
            #     pickle.dump(env, open(os.path.join(CHECKPOINT_DIR, f"best_genome_{self.gen}.pkl"), "wb"))
            #     env.close()
            #     print(f"Best genome found at generation {self.gen} with fitness {reward}")
                
            output.put(total_reward)
            env.close()
        except KeyboardInterrupt:
            env.close()
            exit()
        except Exception as e:
            print(f"Error in fitness function: {e}")
            output.put(0)  # Return zero fitness on error
            env.close()
        
    def _eval_genomes(self, genomes, config):
        idx, genomes = zip(*genomes)
        
        for i in range(0, len(genomes), self.cores):
            output = mp.Queue()
            
            processes = [mp.Process(target=self._fitness_func, args=(genome, config, output)) for genome in
                         genomes[i:i + self.cores]]
            
            [p.start() for p in processes]
            [p.join() for p in processes]
            
            results = [output.get() for p in processes]
            
            for n, r in enumerate(results):
                genomes[i + n].fitness = r
                
    def _run(self, generations):
        win_pop = self.population.run(self._eval_genomes, generations)
        win_gen = self.population.best_genome
        
        # Ensure checkpoint directory exists
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        
        pickle.dump(win_pop, open(os.path.join(CHECKPOINT_DIR, 'winner.pkl'), 'wb'))
        pickle.dump(win_gen, open(os.path.join(CHECKPOINT_DIR, 'real_winner.pkl'), 'wb'))
                
    def main(self):
        self._run(self.gen)
        
if __name__ == "__main__":
    # check last checkpoint
    last_checkpoint = None
    checkpoints = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint-')], key=lambda x: int(x.split('-')[1]))
    if checkpoints:
        last_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoints found. Starting from scratch.")

    t = Trainer(gen=100, cores = 15, resume_from=False, render=False)
    t.main()
        