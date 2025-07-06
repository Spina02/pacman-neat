In this project, I trained a **NEAT (NeuroEvolution of Augmenting Topologies)** agent to play Pac-Man. My primary goal was to evolve a neural network that could develop complex strategies to navigate mazes, evade ghosts, and maximize its score.

The project leverages a **Curriculum Learning** approach to guide the agent's learning process from basic survival mechanics to advanced strategic gameplay.


<details>
<summary><h2>Repository Structure</h2></summary>

The core Pac-Man game logic is adapted from the [PyPacman](https://github.com/AnandSrikumar/PyPacman.git) repository (here my [fork](https://github.com/Spina02/PyPacman.git)). I modified it significantly, including bug fixes and refactoring the game state management to create a stable and efficient environment suitable for machine learning applications.

The key components of this project are:
- **`game/game_env.py`**: This file is the "bridge" between the NEAT algorithm and the Pac-Man game. It implements a Gym-like environment, handling game state, observations, and reward calculations.
- **`neat_model/trainer.py`**: Manages the main training loop, population management, and parallel evaluation of genomes using multiprocessing.
- **`neat_model/run.py`**: Handles the execution and visualization of a single, pre-trained genome.
- **`main.py`**: The main command-line interface (CLI) entry point to orchestrate both training and execution modes.

</details>


<details>
<summary><h2>Highlights</h2></summary>

<details>
<summary><h3>Agent Observation Model</h3></summary>

To make decisions, the model needs to "see" the current state of the game—this observation becomes the input to the neural network. Naturally, the more detailed the observation, the more information the model has to work with, but this also increases computational complexity. Finding the right balance between informativeness and efficiency was a key challenge.

Initially, I started with a "simple" observation vector: 26 elements including:

- **Ghost Data:** Relative positions and "scared" status for all four ghosts.
- **Navigation Aids:** Normalized vectors to the nearest dot and power-up pellet.
- **Game Progress:** The ratio of remaining dots.
- **Local Awareness:** Normalized distances to the nearest walls in four cardinal directions.
- **Internal State:** Pac-Man's power-up status and a one-hot encoding of its last action to prevent oscillations.

However, I soon realized that the model needed a better sense of its immediate surroundings. To address this, I added an **8x8 minimap** centered on Pac-Man, which encodes the positions of walls, dots, power-ups, and ghosts (distinguishing between normal and scared states). This addition brought the total observation size to 90 elements.

</details>

<details>
<summary><h3>Curriculum Learning & Reward Shaping</h3></summary>

Training an agent to master Pac-Man from scratch is very challenging. To overcome this, I designed a structured **Curriculum Learning** path, breaking down the problem into progressively harder tasks, automatically managed by the training script based on the generation number (except for the `feed_forward` tweaking):

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

I implemented this logic in `game/game_env.py`, where the environment switches between a simple `calculate_reward()` and a more complex `_calculate_reward()` depending on the curriculum phase.

</details>
</details>

<details>
<summary><h2>Code Execution</h2></summary>

You can run the code from the command line using the `main.py` script, which is the main entry point for both training and running a pre-trained agent.

<details>
<summary><h3>How to train</h3></summary>


To `train` the model you can use the following command:

```bash
python main.py train [OPTIONS]
```

This command spawns a population of neural networks, evaluates how well they play Pac-Man, and then evolves them over multiple generations. The process repeats, with each new generation (hopefully) getting a little better at the game.

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

</details>
<details>
<summary><h3>How to run</h3></summary>

To `run` a single, previously saved genome, you can use the following command.

```bash
python main.py run [OPTIONS]
```

This command loads and displays the performance of a specific genome. By default, it looks for the best genome found during training.

**Available Arguments (`OPTIONS`):**

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--config` | str | `config` | Path to the NEAT configuration file. **Must be the same** as the one used to train the genome. |
| `--obs_mode`| str | `minimap` | Observation mode (`simple` or `minimap`). **Must match** the one used to train the genome. |
| `--checkpoint` | str | `None` | Specifies the path to a NEAT checkpoint file or a `.pkl` file containing a single genome to run. |
| `--best` | bool | `False` | If set, this flag overrides `--checkpoint` and loads the latest "best" genome saved for the current observation mode (looks for `best_minimap_latest.pkl`). |
| `--max_steps` | int | `None` | Overrides the maximum number of steps for the simulation episode. Useful for testing an agent's longevity. |
| `--debug` | int | `0` | Sets the verbosity level for the console output during the run (from 0 to 3). |
</details>

<summary><h3>Usage Examples</h3></summary>

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
    python main.py run --checkpoint path/to/checkpoint
    ```
</details>

</details>

<details>
<summary><h2>Project Status & Future Work</h2></summary>

The model has demonstrated significant learning progress: it successfully evolved complex strategies for navigating the maze, actively pursuing dots, and dynamically evading ghosts based on their current behavior (chase vs. scatter). The agent can get high scores and clear most of the map, but it hasn't managed to reliably beat a full level yet.


### Technical Optimizations

To handle the long training runs, I heavily **parallelized** the evaluation process. I set up a `worker pool` where each process maintains its own persistent instance of the Pac-Man environment. This approach avoids the overhead of repeatedly initializing the game and significantly speeds up the training time for each generation.

### Future Directions

To help the agent get past its current performance plateau and finally beat a level, I have a few ideas for future work:

1.  **Enhanced Observation Space:**
    The current 8x8 minimap provides local context but may be insufficient for long-term planning. A potential next step is to **increase the minimap size** (e.g., to 12x12 or 16x16) to provide a wider field of view. The ultimate goal would be to give the agent a view of the **entire game grid**. This would mean a much larger input layer for the neural network and, of course, would require more computational power for training.

2.  **Headless, Pygame-Independent Environment:**
    While the current implementation can run without rendering, it still relies on the Pygame backend for game logic updates and surface management. A major optimization would be to build a **"headless-native" environment**. This would mean rewriting the game loop and state management in pure Python/NumPy, without any dependency on Pygame for rendering. This would cut down on computational overhead, allowing for much faster training and making it possible to experiment with bigger and more complex neural networks.

3.  **Advanced Curriculum and Reward Shaping:**
    - Introduce more granular stages to the curriculum, such as gradually increasing ghost speed or intelligence over generations.
    - Experiment with more nuanced reward functions, for instance, by adding a penalty for moving away from the last remaining cluster of dots to encourage "finishing the job".

</details>