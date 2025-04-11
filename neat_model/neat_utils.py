import neat
import os
import pickle

class BestGenomeSaver(neat.reporting.BaseReporter):
    def __init__(self, save_path, filename_prefix='best_genome', generation=0):
        self.save_path = save_path
        self.filename_prefix = filename_prefix
        self.best_fitness_so_far = -float('inf')
        self.current_best_genome = None
        self.current_generation = generation
        # Assicurati che la directory esista
        os.makedirs(self.save_path, exist_ok=True)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        """
        Chiamato dopo la valutazione di una generazione.
        Salva il miglior genoma se migliora il massimo storico.
        """
        if best_genome is None:
             print("Warning: best_genome is None in post_evaluate.")
             genomes = list(population.values())
             if not genomes: return
             genomes.sort(key=lambda g: g.fitness if g.fitness is not None else -float('inf'), reverse=True)
             best_genome = genomes[0]
             if best_genome.fitness is None: return

        if best_genome and best_genome.fitness > self.best_fitness_so_far:
            self.best_fitness_so_far = best_genome.fitness
            self.current_best_genome = best_genome
            generation = self.current_generation

            filename = os.path.join(self.save_path, f"{self.filename_prefix}_gen{generation}_fit{self.best_fitness_so_far:.2f}.pkl")
            print(f"\n--- New best genome found! ---")
            print(f"Generation: {generation}, Fitness: {self.best_fitness_so_far:.2f}")
            print(f"Saving to {filename}")
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(self.current_best_genome, f)
                    
                latest_filename = os.path.join(self.save_path, f"{self.filename_prefix}_latest.pkl")
                with open(latest_filename, 'wb') as f:
                     pickle.dump(self.current_best_genome, f)
                print(f"Updated {latest_filename}")
            except Exception as e:
                print(f"ERROR saving best genome: {e}")
            print("-" * 30)

    def end_generation(self, config, population, species_set):
        pass
    def found_solution(self, config, generation, best):
        pass # Gestito in post_evaluate
    def species_stagnant(self, sid, species):
        pass
    def info(self, msg):
        pass
    def complete_extinction(self):
        pass
    def finish_reporting(self):
         if self.current_best_genome:
             latest_filename = os.path.join(self.save_path, f"{self.filename_prefix}_latest.pkl")
             if not os.path.exists(latest_filename):
                  try:
                      with open(latest_filename, 'wb') as f:
                          pickle.dump(self.current_best_genome, f)
                      print(f"Final best genome saved to {latest_filename}")
                  except Exception as e:
                       print(f"ERROR saving final best genome: {e}")