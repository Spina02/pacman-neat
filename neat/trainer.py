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

# Top-level function to evaluate a single genome.
# It accepts a tuple (genome_id, genome, config, render) and returns (genome_id, total_reward)
def evaluate_genome(args):
    genome_id, genome, config, render = args
    env = None
    try:
        env = game.PacmanEnvironment(render)
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
    except Exception as e:
        print(f"Error in evaluating genome {genome_id}: {e}")
        total_reward = 0
        if env:
            env.close()
        exit(-1)
    return (genome_id, total_reward)

class Trainer:
    def __init__(self, neat_config_file=CONFIG_PATH, gen=1000, cores=14, resume_from=None, render=False):
        self.gen = gen
        self.cores = cores
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
            print(f"Resuming from generation {self.population.generation}")
        else:
            self.population = neat.Population(self.neat_config)

        self.population.add_reporter(neat.StdOutReporter(True))
        prefix = os.path.join(CHECKPOINT_DIR, 'checkpoint-')
        self.population.add_reporter(
            neat.Checkpointer(generation_interval=5, filename_prefix=prefix)
        )
        self.population.add_reporter(neat.StatisticsReporter())

        # Create a pool of processes once
        self.pool = mp.Pool(processes=self.cores)

    def _eval_genomes(self, genomes, config):
        # Prepare a dictionary for quick access
        genome_dict = {genome_id: genome for genome_id, genome in genomes}
        args = [(genome_id, genome, config, self.render) for genome_id, genome in genomes]
        results = self.pool.map(evaluate_genome, args)
        for genome_id, reward in results:
            genome_dict[genome_id].fitness = reward

    def _run(self, generations):
        win_pop = self.population.run(self._eval_genomes, generations)
        win_gen = self.population.best_genome
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        pickle.dump(win_pop, open(os.path.join(CHECKPOINT_DIR, 'winner.pkl'), 'wb'))
        pickle.dump(win_gen, open(os.path.join(CHECKPOINT_DIR, 'real_winner.pkl'), 'wb'))
        # Close the pool at the end of training
        self.pool.close()
        self.pool.join()

    def main(self):
        self._run(self.gen)

if __name__ == "__main__":
    TRAIN = 1
    if not os.path.exists(CHECKPOINT_DIR):
        checkpoints = []
    else:
        checkpoints = sorted(
            [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('checkpoint-')],
            key=lambda x: int(x.split('-')[1])
        )
    last_checkpoint = None
    if checkpoints:
        last_checkpoint = os.path.join(CHECKPOINT_DIR, checkpoints[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")
    else:
        print("No checkpoints found. Starting from scratch.")

    if TRAIN:
        t = Trainer(gen=1000, cores=15, resume_from=last_checkpoint)
    else:
        t = Trainer(gen=1, cores=1, resume_from=last_checkpoint, render=True)
    t.main()
