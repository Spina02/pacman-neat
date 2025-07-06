# Pac-Man NEAT

This project explores the application of **NEAT (NeuroEvolution of Augmenting Topologies)** to train an artificial intelligence agent to play a classic version of Pac-Man. The primary goal is to evolve a neural network capable of developing complex, robust, and effective strategies to navigate mazes, evade ghosts, and maximize score.

The project leverages a sophisticated **Curriculum Learning** approach to guide the agent's learning process from basic survival mechanics to advanced strategic gameplay.

---

## Repository Structure

The core Pac-Man game logic is adapted from the [PyPacman](https://github.com/AnandSrikumar/PyPacman.git) repository. Significant modifications were made to this foundation, including bug fixes and a complete overhaul of the game state management to create a stable and efficient environment suitable for machine learning applications.

The key components of this project are:
- **`game/game_env.py`**: This file is the heart of the project, acting as a bridge between the NEAT algorithm and the Pac-Man game. It implements a Gym-like environment, handling game state, observations, and reward calculations.
- **`neat_model/trainer.py`**: Manages the main training loop, population management, and parallel evaluation of genomes using multiprocessing.
- **`neat_model/run.py`**: Handles the execution and visualization of a single, pre-trained genome.
- **`main.py`**: The main command-line interface (CLI) entry point to orchestrate both training and execution modes.

---

## Technical Highlights

### 1. Agent Observation Model

For the agent to make informed decisions, it must perceive the game's state. The design of this observation vector is a critical trade-off between informational richness and computational complexity.

This project utilizes a hybrid observation model that combines high-level game features with a localized spatial map:

- **Vector-Based State (26 features):**
  - **Ghost Data:** Relative positions and "scared" status for all four ghosts.
  - **Navigation Aids:** Normalized vectors to the nearest dot and power-up pellet.
  - **Game Progress:** The ratio of remaining dots.
  - **Local Awareness:** Normalized distances to the nearest walls in four cardinal directions.
  - **Internal State:** Pac-Man's power-up status and a one-hot encoding of its last action to prevent oscillations.

- **8x8 Minimap (64 features):**
  - A grid-based, Pac-Man-centric view of the immediate surroundings. It encodes the locations of walls, dots, power-ups, and ghosts (differentiating between normal and scared states).

This combined approach provides the agent with both a global strategic context and immediate tactical awareness, resulting in a total input size of **90 neurons**.

### 2. Curriculum Learning & Reward Shaping

Training an agent to master Pac-Man from scratch is a formidable challenge. To overcome this, a structured **Curriculum Learning** path was designed, breaking down the problem into a sequence of progressively harder tasks. This guides the evolution towards meaningful skills at each stage.

The curriculum is divided into the following phases, automatically managed by the training script based on the generation number:

- **Phase 1: Basic Navigation (Generations 0-499)**
  - **Goal:** Learn to move, explore the map, and eat dots.
  - **Environment:** Level 1 (only dots, no ghosts, no power-ups).
  - **Reward:** Simple reward for eating dots, a small penalty for each step (`cost of living`), and large checkpoint bonuses for clearing 25%, 50%, 75%, and 90% of the map.
  - **Architecture:** Feed-Forward Network (`feed_forward = True`).

- **Phase 2: Developing Memory (Generations 500-999)**
  - **Goal:** Evolve more complex, non-reactive strategies.
  - **Environment:** Same as Phase 1.
  - **Architecture:** Recurrent Neural Network (`feed_forward = False`) to allow the emergence of short-term memory.

- **Phase 3: Evasion Training (Generations 1000-1499)**
  - **Goal:** Learn the core skill of evading hostiles.
  - **Environment:** Level 1, but with all ghosts activated in a fixed "chase" mode.
  - **Reward:** Same simple reward function. The primary pressure to evolve comes from the environmental threat.

- **Phase 4: Dynamic Evasion (Generations 1500-1999)**
  - **Goal:** Adapt to changing ghost behaviors.
  - **Environment:** Level 2, where ghosts now alternate between "chase" and "scatter" modes.
  - **Reward:** Still the simple reward function.

- **Phase 5: Full Game Mastery (Generations 2000+)**
  - **Goal:** Master the complete game.
  - **Environment:** Level 3, featuring the full game with dots, power-ups, and dynamic ghosts.
  - **Reward:** A complex, heavily shaped reward function is activated. This includes:
    - Dynamic multipliers for eating dots.
    - Large bonuses for eating power-ups and scared ghosts.
    - An exploration bonus for visiting new tiles.
    - Penalties for inactivity, getting stuck, or being too close to non-scared ghosts.

This entire logic is managed within `game/game_env.py`, switching between a simple `calculate_reward()` and a complex `_calculate_reward()` based on the current curriculum phase.

---

## Code Execution

This project is executed via the command line through the `main.py` script, which acts as the main entry point for both training a new model and running a pre-existing one.

### How to train

The `train` mode initiates the NEAT evolutionary process. It creates, evaluates, and evolves a population of neural networks over a specified number of generations.

**Base Command:**
```bash
python main.py train [OPTIONS]
```

**Available Arguments (`OPTIONS`):**

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--render` | bool | `False` | If set to `True`, it displays the game window during training. This is useful for debugging but significantly slows down the process. |
| `--config` | str | `config` | Specifies the path to the NEAT configuration file to be used. |
| `--generations` | int | `2000` | Defines the total number of generations the training should run for. |
| `--checkpoint_dir`| str | `checkpoints`| Specifies the directory where checkpoint files will be saved. |
| `--checkpoint` | str | `None` | Path to a specific checkpoint file to resume the session from. If not provided, the script will automatically look for and load the latest checkpoint in the `checkpoint_dir`. |
| `--obs_mode` | str | `minimap` | Selects the observation type the agent receives. Valid options are `simple` or `minimap`. This **must match** the `num_inputs` setting in the config file. |
| `--cores` | int | `15` | Sets the number of CPU cores to use for parallel evaluation of genomes. Higher values accelerate training. |
| `--reset` | bool | `False` | If `True`, it will delete all existing checkpoints and best genomes for the current `obs_mode` before starting a new training session from scratch. |

### How to run

The `run` mode allows for the visualization of the performance of a single, previously saved genome.

**Base Command:**
```bash
python main.py run [OPTIONS]
```

**Available Arguments (`OPTIONS`):**

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | str | `config` | Path to the NEAT configuration file. **Must be the same** as the one used to train the genome. |
| `--obs_mode`| str | `minimap` | Observation mode (`simple` or `minimap`). **Must match** the one used to train the genome. |
| `--checkpoint` | str | `None` | Specifies the path to a NEAT checkpoint file or a `.pkl` file containing a single genome to run. |
| `--best` | bool | `False` | If set, this flag overrides `--checkpoint` and loads the latest "best" genome saved for the current observation mode (looks for `best_minimap_latest.pkl`). |
| `--max_steps` | int | `None` | Overrides the maximum number of steps for the simulation episode. Useful for testing an agent's longevity. |
| `--debug` | int | `0` | Sets the verbosity level for the console output during the run (from 0 to 3). |

#### Usage Examples

1. **Start a new training session:**

    ```bash
    python main.py train --reset 1
    ```

2. **Continue training session:**

    ```bash
    python main.py train
    ```

3.  **Run the best genome found so far:**
    ```bash
    python main.py run --best
    ```

4.  **Run the best genome from a specific checkpoint:**
    ```bash
    python main.py run --checkpoint checkpoints/checkpoint-minimap-1527
    ```

## Project Status & Future Work

The model has undergone extensive training through the defined curriculum and has demonstrated significant learning progress. It has successfully evolved complex strategies for navigating the maze, actively pursuing dots, and dynamically evading ghosts based on their current behavior (chase vs. scatter). The agent consistently achieves high scores, clearing a large portion of the map.

However, the agent has not yet been able to reliably complete an entire level. A common failure mode occurs in the late-game, where the agent, despite its advanced evasive maneuvers, can be trapped in complex cornering situations by the remaining ghosts. This suggests that while its tactical, short-term decision-making is highly evolved, it may lack the long-term strategic foresight to avoid endgame traps.

### Technical Optimizations

To facilitate these long training runs, the evaluation process has been heavily optimized:
- **Parallelization:** The fitness evaluation of genomes is parallelized across multiple CPU cores. A `worker pool` architecture is used, where each process maintains its own persistent instance of the Pac-Man environment. This minimizes the overhead of repeated environment initializations, leading to a significant speedup in training time per generation.

### Future Directions

To overcome the current performance plateau and push the agent towards completing the level, several future research avenues are being considered:

1.  **Enhanced Observation Space:**
    - The current 8x8 minimap provides local context but may be insufficient for long-term planning. A potential next step is to **increase the minimap size** (e.g., to 12x12 or 16x16) to provide a wider field of view. The ultimate goal would be to provide the agent with a representation of the **entire game grid**, which would require a substantial increase in the network's input layer and, consequently, more computational resources for training.

2.  **Headless, Pygame-Independent Environment:**
    - While the current implementation can run without rendering, it still relies on the Pygame backend for game logic updates and surface management. A major optimization would be to develop a **"headless-native" environment**. This would involve re-implementing the core game loop and state management purely in Python/NumPy, completely independent of Pygame's rendering engine. This would drastically reduce computational overhead, allowing for significantly faster training cycles and enabling the exploration of much larger and more complex neural network architectures.

3.  **Advanced Curriculum and Reward Shaping:**
    - Introduce more granular stages to the curriculum, such as gradually increasing ghost speed or intelligence over generations.
    - Experiment with more nuanced reward functions, for instance, by adding a penalty for moving away from the last remaining cluster of dots to encourage "finishing the job".
