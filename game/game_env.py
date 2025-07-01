import os
import pygame
import sys
import numpy as np

pacman_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pacman"))
if (pacman_path not in sys.path):
    sys.path.insert(0, pacman_path)

from src.game.state_management import GameState
from src.game.event_management import EventHandler
from src.gui.screen_management import ScreenManager
from src.utils.coord_utils import get_idx_from_coords, get_float_idx_from_coords
from src.configs import CELL_SIZE, NUM_ROWS, NUM_COLS

#?--------------------------------------------------------------
#?                    Environment class
#?--------------------------------------------------------------

class PacmanEnvironment:
    """
    Gym-like environment for Pacman.
    Handles initialization, stepping with an action,
    returning observations and rewards, etc.
    """

    def __init__(self, render, observation_mode='minimap'):
        self.last_action = None
        self.game_state = None
        self.screen = None
        self.event_handler = None
        self.all_sprites = None
        self.screen_manager = None
        self.prev_points = 0
        self.stuck_counter = 0
        # self.prev_dist_to_closest_dot = float('inf')
        self.render_enabled = render
        self.observation_mode = observation_mode
        self.px = 0 # Pacman grid column index
        self.py = 0 # Pacman grid row index
        self.visits = None
        self.neg_count = 0
        self.current_gen = 0
        self.steps_since_last_dot = 0
        self.last_action = None
        self.eaten_ghosts = 0
        self.n_opposite = 0
        self.pacman_pos = None
        self.tot_visited = 0
        self.debug = 0
        self.max_reached = False
        self.ghost_order = ['blinky', 'pinky', 'inky', 'clyde']
        self.EASY_GEN = [0, 500, 750, 1000] # Generations at which each ghost is added
        self.MAX_EPISODE_STEPS = 5000 # Default max steps per episode
        self.wall_distance_map = None # Pre-computed wall distances
        self.wall_code = None  # Will be set in reset()
        self.dot_code = None  # "dot"  tile code
        self.power_code = None  # "power" tile code
        # --------------------------------------------------------------
        # Cached dynamic item positions (initialised in reset())
        # --------------------------------------------------------------
        self.dot_positions = None   # np.ndarray of dot coordinates (rows, cols)
        self.power_positions = None # np.ndarray of power-up coordinates

        # Pre-compute reciprocals of grid dimensions to replace repeated division
        self._inv_num_rows = 1.0 / NUM_ROWS
        self._inv_num_cols = 1.0 / NUM_COLS
        self.progress_checkpoints = {}

    #?--------------------------------------------------------------
    #?                    Initialization functions
    #?--------------------------------------------------------------

    def reset(self):
        """
        Initialize or re-initialize the game state,
        returning the initial observation.
        """
        # Setup for headless mode BEFORE pygame.init()
        if not self.render_enabled:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                
        if not pygame.get_init():
            pygame.init()
            if not self.render_enabled:
                # Setting a dummy display might still be needed by some pygame parts
                pygame.display.set_mode((1, 1), pygame.NOFRAME)
        
        if self.render_enabled:
            self.screen = pygame.display.set_mode((1024, 768))
        else:
            self.screen = pygame.Surface((1024, 768)) # Create a surface for drawing if needed
            
        self.game_state = GameState()
        self.game_state.sound_enabled = False
        # # If rendering is enabled but screen wasn't created (after headless), create it
        # if self.render_enabled and self.screen is None:
        #      self.screen = pygame.display.set_mode((1024, 768))
        # # If not rendering and screen exists (from previous render run), make it a surface
        # elif not self.render_enabled and self.screen is not None and not isinstance(self.screen, pygame.Surface):
        #      self.screen = pygame.Surface((1024, 768))
        # # Ensure screen is a valid surface if not rendering
        # elif not self.render_enabled and self.screen is None:
        #      self.screen = pygame.Surface((1024, 768))

        self.event_handler = EventHandler(self.screen, self.game_state)
        self.all_sprites = pygame.sprite.Group()

        self.prev_points = 0
        self.neg_count = 0
        # self.prev_dist_to_closest_dot = float('inf')
        self.stuck_counter = 0
        self.steps_since_last_dot = 0
        self.last_action = None
        self.eaten_ghosts = 0
        self.n_opposite = 0
        self.tot_visited = 0

        # ScreenManager handles sprite creation internally based on game_state
        self.screen_manager = ScreenManager(self.screen, self.game_state, self.all_sprites)

        # Initialize Pacman's grid position
        self._update_pacman_grid_position()

        # Update sprites once to set initial positions based on game_state
        self.all_sprites.update(0)

        rows, cols = self.game_state.level_matrix_np.shape
        self.visits = np.zeros((rows, cols), dtype=int)
        
        # --------------------------------------------------------------
        # Cache tile codes once per reset (micro-optimisation)
        # --------------------------------------------------------------
        self.wall_code  = self.game_state.tile_encoding.get("wall", 0)
        self.dot_code   = self.game_state.tile_encoding.get("dot", 1)
        self.power_code = self.game_state.tile_encoding.get("power", 2)
        
        # Pre-compute wall distances for the whole level grid
        self._precompute_wall_distances()
        
        # Cache initial dot / power-up positions to avoid np.argwhere in hot path
        grid = self.game_state.level_matrix_np
        self.dot_positions   = np.argwhere(grid == self.dot_code)
        self.power_positions = np.argwhere(grid == self.power_code)
        
        if self.current_gen < self.EASY_GEN[0]:
            self.MAX_EPISODE_STEPS = 2000
            
        for i in range(len(self.ghost_order)):
            if self.current_gen < self.EASY_GEN[i]:
                self.game_state.no_ghosts[self.ghost_order[i]] = True
            else:
                self.game_state.no_ghosts[self.ghost_order[i]] = False
                
        #START_MAX_STEPS = 2000
        #FINAL_MAX_STEPS = 4000
        #GENS_TO_REACH_FINAL = 300 # Reach final limit after 200 generations
        #
        #if self.current_gen < GENS_TO_REACH_FINAL:
        #    self.MAX_EPISODE_STEPS = int(START_MAX_STEPS + (FINAL_MAX_STEPS - START_MAX_STEPS) * (self.current_gen / GENS_TO_REACH_FINAL))
        #else:
        #    self.MAX_EPISODE_STEPS = FINAL_MAX_STEPS
        
        # Use cached dot positions for efficiency
        self.total_dots = len(self.dot_positions)
        self.remaining_dots = self.total_dots
        
        self.progress_checkpoints = {
            0.9: False,
            0.75: False,
            0.5: False,
            0.25: False
        }
        
        # Get initial observation based on the configured mode
        obs = self.get_observation(mode=self.observation_mode)

        return obs

    #?--------------------------------------------------------------
    #?                        Step function
    #?--------------------------------------------------------------

    def step(self, action):
        """
        Advance one frame with the specified action.
        Args:
            action (int): Action chosen by NEAT: 0=left, 1=right, 2=up, 3=down
        Returns:
            observation (list or np.array)
            reward (float)
            done (bool)
            info (dict)
        """
        current_points = self.game_state.points
        
        # 1) Translate action integer into game_state.direction
        if action == 0:
            self.game_state.direction = "l"
        elif action == 1:
            self.game_state.direction = "r"
        elif action == 2:
            self.game_state.direction = "u"
        elif action == 3:
            self.game_state.direction = "d"
        else:
            print("Invalid action:", action)

        # 2) Update one frame
        reward = 0.0
        done = False
        # Use cached dot code
        dot_code = self.dot_code

        #? replication of "GameRun.main()" loop but only for 1 iteration
        # Handle Pygame events (important even in headless for QUIT signals etc.)
        for event in pygame.event.get():
            # Pass manual=False as NEAT controls direction
            self.event_handler.handle_events(event, manual=False)

        # Check time-based game events (ghost mode changes, powerup expiry)
        self.event_handler.check_frame_events()

        # Update game logic (sprite movements, collisions)
        dt = 1
        self.all_sprites.update(dt)
        
        # Update Pacman's grid position AFTER movement
        self._update_pacman_grid_position()

        # --------------------------------------------------------------
        # Refresh cached dot / power-up coordinates after state update
        # --------------------------------------------------------------
        self._refresh_item_positions()

        # Get the current observation
        obs = self.get_observation(mode=self.observation_mode)

        # Using cached arrays we can compute remaining dots quickly
        prev_remaining_dots = self.remaining_dots
        self.remaining_dots = len(self.dot_positions)
        dots_eaten_this_step = prev_remaining_dots - self.remaining_dots
        
        # Determine if done
        done = self.game_state.is_pacman_dead or self.game_state.level_complete
        
        # Update steps since last dot counter
        if dots_eaten_this_step > 0:
            self.steps_since_last_dot = 0
        else:
            self.steps_since_last_dot += 1
            
        reward = self.calculate_reward(current_points, obs)
        
        self.last_action = action
        
        # Safety check
        rows, cols = self.visits.shape
        if 0 <= self.py < rows and 0 <= self.px < cols:
            self.visits[self.py, self.px] += 1
        
        # Max steps check
        timeout_penalty = -100
        if not done and self.game_state.step_count > self.MAX_EPISODE_STEPS:
            reward += timeout_penalty
            # End the episode due to timeout
            self.max_reached = True
            done = True

        self.prev_points = self.game_state.points

        # Render if enabled
        if self.render_enabled:
            self.screen.fill((0, 0, 0))
            self.screen_manager.draw_screens() # Draws level, scores
            self.all_sprites.draw(self.screen) # Draws Pacman, Ghosts
            pygame.display.flip()

        self.last_action = action
        self.game_state.step_count += 1

        info = {} # Placeholder for additional info if needed later
        return obs, reward, done, info


    # Helper function to pre-compute wall distances for the entire grid
    def _precompute_wall_distances(self):
        """
        Calculates the normalized distance to the nearest wall in 4 directions
        for every non-wall cell in the grid. Results are stored in a lookup table.
        This is done once per level reset to optimize the get_observation method.
        """
        grid = self.game_state.level_matrix_np
        rows, cols = grid.shape
        wall_code = self.wall_code  # Cached value
        self.wall_distance_map = np.zeros((rows, cols, 4), dtype=np.float32)

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == wall_code:
                    continue  # Distances are 0 for wall cells

                # dist_left (Equivalent to distance_to_wall(r, c, 0, -1))
                dist = 0
                for i in range(c - 1, -1, -1):
                    if grid[r, i] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 0] = dist / cols

                # dist_right (Equivalent to distance_to_wall(r, c + 1, 0, 1))
                dist = 0
                # Original logic started scan from c + 2
                for i in range(c + 2, cols):
                    if grid[r, i] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 1] = dist / cols

                # dist_up (Equivalent to distance_to_wall(r, c, -1, 0))
                dist = 0
                for i in range(r - 1, -1, -1):
                    if grid[i, c] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 2] = dist / rows

                # dist_down (Equivalent to distance_to_wall(r + 1, c, 1, 0))
                dist = 0
                # Original logic started scan from r + 2
                for i in range(r + 2, rows):
                    if grid[i, c] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 3] = dist / rows

    # Helper function to update Pacman's grid position
    def _update_pacman_grid_position(self):
        """ Gets Pacman's current pixel coords and converts them to grid indices. """
        if self.game_state.pacman_rect and self.game_state.start_pos:
            px_pixel, py_pixel = self.game_state.pacman_rect[:2]
            start_x, start_y = self.game_state.start_pos
            cell_w, cell_h = CELL_SIZE
            
            # Convert pixel coordinates to grid indices
            grid_y, grid_x = get_idx_from_coords(px_pixel, py_pixel, start_x, start_y, cell_w)
            self.py_float, self.px_float = get_float_idx_from_coords(px_pixel, py_pixel, start_x, start_y, cell_w)
            # Ensure indices are within bounds
            self.py = max(0, min(NUM_ROWS - 1, grid_y))
            self.px = max(0, min(NUM_COLS - 1, grid_x))
        else:
            # Default or error handling if state is not ready
            self.py = 0
            self.px = 0
            print("Warning: Pacman rect or start pos not available for grid calculation.")


    #?--------------------------------------------------------------
    #?                    Observation function
    #?--------------------------------------------------------------

    def get_observation(self, mode="minimap"):
        """
        Returns an observation representation of the current game state.

        Args:
            mode (str): Observation type to return. Currently supported: "simple", "minimap".
                        Defaults to "simple".

        Returns:
            if mode == "simple":
                numpy.ndarray: Observation vector (total 24 elements) containing:
                    - Pacman's normalized grid position (row, col) [2 elements]
                    - Relative normalized positions of up to 4 ghosts (fixed order: 'blinky', 'pinky', 'inky', 'clyde') [8 elements]
                    - Ghost scared bits (1.0 if scared, 0.0 otherwise) [4 elements]
                    - Vector (normalized) from Pacman to the nearest dot [2 elements]
                    - Vector (normalized) from Pacman to the nearest powerup [2 elements]
                    - Number of remaining dots [1 element]
                    - Normalized distances to walls in 4 directions (left, right, up, down) [4 elements]
                    - Pacman's power-up state (1.0 if active, 0.0 otherwise) [1 element]
            if mode == "minimap":
                numpy.ndarray: Observation vector (total 24 elements + 64 minimap elements) containing:
        """
        # Use local variables for frequently accessed attributes
        game_state = self.game_state
        grid = game_state.level_matrix_np
        pacman_r_grid, pacman_c_grid = self.py, self.px
        pacman_r_float, pacman_c_float = self.py_float, self.px_float

        if mode == "simple" or mode == "minimap":
            pacman_grid = np.array([pacman_r_grid, pacman_c_grid])

            # Normalize Pacman position
            self.pacman_pos = np.array([pacman_r_float, pacman_c_float])

            # (2) Relative positions of ghosts (GRID BASED)
            ghost_rel = -np.ones(8)  # 4 ghosts * 2 coords
            if not all(game_state.no_ghosts.values()):
                ghosts = game_state.ghosts  # dict: ghost_name -> (gx, gy) in pixels
                for i, name in enumerate(self.ghost_order):
                    if name in ghosts:
                        gx, gy = ghosts[name]
                        rel_r = (gx - pacman_grid[0]) / NUM_ROWS
                        rel_c = (gy - pacman_grid[1]) / NUM_COLS
                        ghost_rel[i*2]     = rel_r
                        ghost_rel[i*2 + 1] = rel_c
            # If the ghost doesn't exist or isn't placed, leave (-1,-1)

            # (3) Ghost scared bits
            ghost_scared_bits = np.array([1.0 if game_state.scared_ghosts[game_state.ghost_encoding[name]] else 0.0 for name in self.ghost_order], dtype=np.float32)

            # (4) Compute the vector to the nearest dot.

            # Use cached dot positions (updated each step)
            dots = self.dot_positions
            if dots.size > 0:
                # Compute squared distances (cheaper than sqrt) – sqrt not needed for argmin
                d_sq = np.sum((dots - self.pacman_pos) ** 2, axis=1)
                idx = np.argmin(d_sq)
                nearest_dot_grid = dots[idx] # Grid coords [row, col] of nearest dot
                # Vector from Pacman to dot, normalized
                diff = nearest_dot_grid - self.pacman_pos
                closest_dot_vec = np.array([diff[0] * self._inv_num_rows, diff[1] * self._inv_num_cols])
            else:
                closest_dot_vec = np.array([-1, -1])

            # (5) Nearest powerup (if any)
            # Use cached power-up positions
            powerups = self.power_positions
            if powerups.size > 0:
                d_sq = np.sum((powerups - self.pacman_pos) ** 2, axis=1)
                idx = np.argmin(d_sq)
                nearest_powerup_grid = powerups[idx]
                diff_p = nearest_powerup_grid - self.pacman_pos
                closest_powerup_vec = np.array([diff_p[0] * self._inv_num_rows, diff_p[1] * self._inv_num_cols])
            else:
                closest_powerup_vec = np.array([-1, -1])
                                
            # (6) Remaining dots
            remaining_dots = [self.remaining_dots/self.total_dots]

            # (7) Get pre-computed distances to the nearest wall in 4 directions
            r, c = pacman_grid.astype(int)
            wall_dists = self.wall_distance_map[r, c]

            # (8) Pacman powerup status
            power_state = np.array([game_state.is_pacman_powered])
            
            # one hot encoding of last action
            last_action = np.zeros(4)
            if self.last_action is not None:
                last_action[self.last_action] = 1.0
            
            # Combine everything into one observation vector
            observation = np.concatenate([
                ghost_rel,            # 8 -> [0, 7]
                ghost_scared_bits,    # 4 -> [8, 11]
                closest_dot_vec,      # 2 -> [12, 13]
                closest_powerup_vec,  # 2 -> [14, 15]
                remaining_dots,       # 1 -> [16]
                wall_dists,           # 4 -> [17, 20]
                power_state,          # 1 -> [21]
                last_action           # 4 -> [22, 25]
            ]).astype(np.float32)     # Total: 26 elements
            
            if mode == "simple":
                return observation

        #elif mode == "minimap":
            minimap_size = 8
            radius = minimap_size // 2

            # Define encoding for different elements in the minimap
            encoding = {
                "wall": -1.0,
                "dot": 0.5,
                "power": 0.75,
                "void": 0.0, # Empty space Pacman can move into
                "elec": -1.0, # Treat electric fence like a wall
                "ghost_normal": -0.75,
                "ghost_scared": 1.0,
                "out_of_bounds": -1.0 # Treat out of bounds like a wall
            }

            # Initialize minimap with 'out_of_bounds' value
            minimap = np.full((minimap_size, minimap_size), encoding["out_of_bounds"], dtype=np.float32)

            # Get Pacman's current grid position (already updated)
            pacman_r, pacman_c = self.py, self.px

            # Get the full game grid and tile decodings
            tile_code_to_name = game_state.tile_decoding # Map code (0,1,2...) back to name ('wall', 'dot'...)
            max_rows, max_cols = grid.shape

            # Fill the minimap with static elements (walls, dots, etc.)
            for r_mini in range(minimap_size):
                for c_mini in range(minimap_size):
                    # Calculate corresponding world grid coordinates. No +1 for even-sized map.
                    world_r = pacman_r + (r_mini - radius)
                    world_c = pacman_c + (c_mini - radius)

                    # Check if the world coordinates are within the game grid boundaries
                    if 0 <= world_r < max_rows and 0 <= world_c < max_cols:
                        tile_code = grid[world_r, world_c]
                        tile_name = tile_code_to_name.get(tile_code, "void") # Default to void if code unknown
                        minimap[r_mini, c_mini] = encoding.get(tile_name, encoding["void"]) # Use encoded value
                    # else: coordinates are out of bounds, keep the default 'out_of_bounds' value

            # Overlay ghosts onto the minimap
            ghosts_pixel_coords = game_state.ghosts # dict: name -> (pixel_x, pixel_y)
            ghost_name_to_idx = game_state.ghost_encoding # Map name to index (0-3)

            for name in self.ghost_order:
                if name in ghosts_pixel_coords and name in ghost_name_to_idx:
                    ghost_r, ghost_c = ghosts_pixel_coords[name]
                    
                    # Calculate relative position to Pacman in grid terms
                    rel_r = ghost_r - pacman_r
                    rel_c = ghost_c - pacman_c

                    # Check if the ghost is within the minimap radius
                    if abs(rel_r) < radius and abs(rel_c) < radius:
                        # Calculate minimap indices
                        r_mini = rel_r + radius
                        c_mini = rel_c + radius

                        # Check if the ghost is scared
                        ghost_idx = ghost_name_to_idx[name]
                        is_scared = game_state.scared_ghosts[ghost_idx]

                        # Set the minimap cell value (ghosts overwrite static elements)
                        minimap[r_mini, c_mini] = encoding["ghost_scared"] if is_scared else encoding["ghost_normal"]

            # # print the matrix for debugging
            # for row in minimap:
            #     print(" ".join(f"{val:.2f}\t" for val in row))
            # print("\n") # Newline for separation

            # Flatten the 2D minimap into a 1D vector for NEAT
            observation = np.concatenate([observation, minimap.flatten()])
            #print(f"Minimap Obs Shape: {observation.shape}")
            
            return observation

        else:
            print(f"Warning: Unknown observation mode '{mode}'. Returning empty array.")
            return np.array([])

    #?--------------------------------------------------------------
    #?                      Reward function
    #?--------------------------------------------------------------

    def calculate_reward(self, previous_points, current_observation):
        """
        Calculates the reward for the current step.
        Args:
            previous_points (int): Score before the current step.
            current_observation (np.array or None): The observation vector
                from simple mode IF self.observation_mode is 'simple'. Otherwise None.
                Used for features like wall distance if needed.
        """
        reward = 0.0
        
        # Use local variables for frequently accessed attributes
        game_state = self.game_state
        current_points = game_state.points
        action_map = {"l": 0, "r": 1, "u": 2, "d": 3}
        current_action_int = action_map.get(game_state.direction)
        
        if self.debug >= 3:
            print(f"Step: {game_state.step_count}, Pacman position: ({self.py}, {self.px}), Action: {current_action_int}")


        # --- One-time Checkpoint Bonuses ---
        eaten_dots = self.total_dots - self.remaining_dots
        eaten_ratio = eaten_dots / self.total_dots if self.total_dots > 0 else 0

        # Check from highest to lowest to award only one bonus per step
        if eaten_ratio > 0.9 and not self.progress_checkpoints[0.9]:
            bonus = 2000.0
            reward += bonus
            self.progress_checkpoints[0.9] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 90%! +{bonus}")
            
        elif eaten_ratio > 0.75 and not self.progress_checkpoints[0.75]:
            bonus = 750.0
            reward += bonus
            self.progress_checkpoints[0.75] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 75%! +{bonus}")
            
        elif eaten_ratio > 0.5 and not self.progress_checkpoints[0.5]:
            bonus = 500.0
            reward += bonus
            self.progress_checkpoints[0.5] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 50%! +{bonus}")
        elif eaten_ratio > 0.25 and not self.progress_checkpoints[0.25]:
            bonus = 250.0
            reward += bonus
            self.progress_checkpoints[0.25] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 25%! +{bonus}")
        
        # 1. Cost of Living
        COST_OF_LIVING = -0.1 #?
        reward += COST_OF_LIVING
        if self.debug >= 3: print(f"cost of living: {COST_OF_LIVING}")

        # 2. Reward for Points Gained
        # multiplier proportional to eaten_dots/total_dots
        SCORE_MULTIPLIER = 1 + 8 * (eaten_dots / self.total_dots)
        points_gain = current_points - previous_points
        
        last_reward = reward
        tmp = ""

        # 3. Specific Event Bonuses
        DOT_EATEN_BONUS = 10.0
        POWER_PELLET_EATEN_BONUS = 20.0
        GHOST_EATEN_BONUS = 50.0

        if points_gain == 10: # Pacman ate a standard dot
            reward += 10 * SCORE_MULTIPLIER
            reward += DOT_EATEN_BONUS
            tmp = "dot"
        elif points_gain == 15: # Pacman ate a power pellet
            self.eaten_ghosts = 0
            reward += 75 * SCORE_MULTIPLIER
            reward += POWER_PELLET_EATEN_BONUS
            tmp = "power up"
        elif points_gain == 25: # Pacman ate a ghost
            self.eaten_ghosts += 1
            reward += 200 * self.eaten_ghosts # * SCORE_MULTIPLIER
            reward += GHOST_EATEN_BONUS
            tmp = "ghost"
        elif points_gain > 0:
            reward += points_gain * SCORE_MULTIPLIER
            tmp = "other"
        if points_gain > 0 and self.debug >= 3: print(f"Pacman ate a {tmp}, reward: {reward - last_reward}")

        # 4. Penalty for Death
        DEATH_PENALTY = -1000.0
        if game_state.is_pacman_dead:
            reward += DEATH_PENALTY
            if self.debug >= 3: print(f"Pacman died, reward: {DEATH_PENALTY}")

        # 5. Bonus for Level Completion
        LEVEL_COMPLETE_BONUS = 10000.0 # Large bonus for completing the level
        if game_state.level_complete:
            reward += LEVEL_COMPLETE_BONUS
            if self.debug >= 3: print(f"Pacman completed the level, reward: {LEVEL_COMPLETE_BONUS}")

        # 6. Penalty for Getting Stuck Against Wall
        if current_observation is not None:
            STUCK_PENALTY_FACTOR = -0.2
            STUCK_PENALTY_CAP = -2 # Limit penalty increase
            STUCK_LIMIT = 200 # Limit for stuck penalty
            #MIN_STUCK_COUNT_FOR_PENALTY = 4 # Minimum stuck count to start penalty

            dir_decode = {"l": 0, "r": 1, "u": 2, "d": 3}
            wall_dists = current_observation[17:21]
            current_dir_idx = dir_decode.get(game_state.direction)

            if current_dir_idx is not None and len(wall_dists) > current_dir_idx:
                wall_dist = wall_dists[current_dir_idx]
                if wall_dist < 1e-6: # Pacman is stuck against a wall
                    self.stuck_counter += 1
                    if self.stuck_counter > STUCK_LIMIT:
                        stuck_penalty = 0
                    else:
                        stuck_penalty = max(STUCK_PENALTY_FACTOR * self.stuck_counter, STUCK_PENALTY_CAP)
                    reward += stuck_penalty
                    if self.debug >= 3: print(f"Pacman stuck against wall, reward: {stuck_penalty}")
                else:
                    self.stuck_counter = 0
            else:
                self.stuck_counter = 0   
                
        # 7. Reward/Penalty for Ghost Interaction
        GHOST_INTERACTION_FACTOR = 1
        if not all(game_state.no_ghosts.values()):
            gh_reward = 0.0
            # Reuse pre-calculated float coordinates of Pacman
            pacman_r, pacman_c = self.py_float, self.px_float
            ghosts_pixel_coords = game_state.ghosts
            ghost_name_to_idx = game_state.ghost_encoding

            # Define a proximity threshold in grid units
            proximity_threshold_grid = 7.0 # for squared comparison
            proximity_threshold_grid_sq = 49.0 # for squared comparison
                    
            for name in self.ghost_order:
                if name in ghosts_pixel_coords and name in ghost_name_to_idx:
                    ghost_r, ghost_c = ghosts_pixel_coords[name]
                    if self.debug >= 3: print(f"Ghost {name} position: ({ghost_r}, {ghost_c}), pacman position: ({pacman_r}, {pacman_c})")
                    # Calculate grid distance (Euclidean)
                    dist_grid_sq = (ghost_r - pacman_r)**2 + (ghost_c - pacman_c)**2


                    if dist_grid_sq < proximity_threshold_grid_sq and dist_grid_sq > 0: # Avoid division by zero or self-collision case
                        # Closeness factor: 1 when very close, 0 at threshold
                        closeness_factor = (proximity_threshold_grid - np.sqrt(dist_grid_sq)) / proximity_threshold_grid

                        # Check if the ghost is scared
                        ghost_idx = ghost_name_to_idx[name]
                        is_scared = game_state.scared_ghosts[ghost_idx]

                        # Reward approaching scared ghosts, penalize approaching normal ghosts
                        sign = 1.0 if is_scared else -1.0
                        # Use squared closeness factor for stronger effect when very close
                        gh_points = sign * GHOST_INTERACTION_FACTOR * closeness_factor**2
                        gh_reward += gh_points
                        
                        if self.debug >= 3: print(f"Pacman close to ghost {name} ({"scared" if is_scared else "not scared"}), distance: {dist_grid_sq}, reward: {gh_points}")

            reward += gh_reward

        # 8. Penalty for Immediate Reversal (Anti-Oscillation)
        REVERSAL_PENALTY = -0.15
        MAX_REVERSAL_PENALTY = 3 # Maximum penalty for reversal
        last_reward = reward
        if self.last_action is not None and current_action_int is not None:
            # Check if actions are opposite (l<->r or u<->d)
            is_opposite = abs(current_action_int - self.last_action) == 1 and current_action_int // 2 == self.last_action // 2
            # Apply penalty only if no points were gained (didn't just eat something)
            if is_opposite and points_gain <= 0:
                self.n_opposite += 1
                reward += REVERSAL_PENALTY*min(self.n_opposite, MAX_REVERSAL_PENALTY)
                if self.debug >= 3: print(f"Pacman reversed direction, reward: {reward - last_reward}")
            else:
                self.n_opposite = 0
                
        # 9. Exploration / Visit Penalty
        VISIT_BONUS_FIRST =        0.5 # Bonus for first visit
        VISIT_PENALTY_THRESHOLD =    4 # Start penalizing later
        VISIT_PENALTY_FACTOR =   -0.25 # Penalty factor for each visit over threshold
        VISIT_PENALTY_CAP =         -4 # Cap penalty to avoid excessive negative rewards

        rows, cols = self.visits.shape
        current_row, current_col = int(self.py), int(self.px)
        if 0 <= current_row < rows and 0 <= current_col < cols:
            visit_count = self.visits[current_row, current_col]

            last_reward = reward
            if visit_count == 0: self.tot_visited += 1
            if visit_count >= 0 and visit_count <= 3:
                if self.n_opposite == 0:
                    reward += VISIT_BONUS_FIRST
            elif visit_count > VISIT_PENALTY_THRESHOLD:
                visit_penalty = VISIT_PENALTY_FACTOR * (visit_count - VISIT_PENALTY_THRESHOLD)
                reward += max(VISIT_PENALTY_CAP, visit_penalty)
            if self.debug >= 3: print(f"Pacman visited {[current_row, current_col]} (count = {visit_count}), reward: {reward - last_reward}")
                 
        # 10. Penalty for not eating dots
        NO_DOT_PENALTY_START_STEP =  100 # Start penalizing after 50 steps without eating
        NO_DOT_PENALTY_FACTOR =    -0.05 # Penalty per step after threshold
        NO_DOT_PENALTY_CAP =          -1 # Maximum penalty per step for this

        last_reward = reward
        if self.steps_since_last_dot > NO_DOT_PENALTY_START_STEP:
            no_dot_penalty = NO_DOT_PENALTY_FACTOR * (self.steps_since_last_dot - NO_DOT_PENALTY_START_STEP)
            reward += max(NO_DOT_PENALTY_CAP, no_dot_penalty)
            if self.debug >= 3: print(f"Pacman hasn't eaten a dot in {self.steps_since_last_dot} steps, reward: {reward - last_reward}")
      
        if self.debug >= 3: print(f"Total reward: {reward:.3f}\n")
        
        return reward


    def close(self):
        """Clean up resources properly"""
        # Stop all sounds (if any)
        if hasattr(self, 'game_state') and self.game_state.sound_enabled:
            if pygame.mixer.get_init():
                pygame.mixer.stop()
                pygame.mixer.quit()
        
        # If we're in a worker process this will be handled by the main process
        if not hasattr(self, '_is_worker') or not self._is_worker:
            if pygame.get_init():
                pygame.quit()

    #?--------------------------------------------------------------
    #?            Helper to refresh dot / power-up positions
    #?--------------------------------------------------------------

    def _refresh_item_positions(self):
        """Removes coordinates that no longer contain dot/power-up to keep caches updated."""
        grid = self.game_state.level_matrix_np

        if self.dot_positions is not None and self.dot_positions.size > 0:
            mask = grid[self.dot_positions[:, 0], self.dot_positions[:, 1]] == self.dot_code
            self.dot_positions = self.dot_positions[mask]

        if self.power_positions is not None and self.power_positions.size > 0:
            mask = grid[self.power_positions[:, 0], self.power_positions[:, 1]] == self.power_code
            self.power_positions = self.power_positions[mask]

if __name__ == "__main__":
    import pygame

    # --- Test with Minimap ---
    print("Testing Minimap Observation Mode...")
    env_minimap = PacmanEnvironment(render=True, observation_mode='minimap')
    env_minimap.current_gen = 9999
    obs_minimap = env_minimap.reset()
    print(f"Initial Minimap Observation Shape: {obs_minimap.shape}")
    # print("Initial Minimap Observation:\n", obs_minimap.reshape(8, 8)) # Print reshaped for readability
    done_minimap = False
    total_reward_minimap = 0.0
    last_action_minimap = None

    # --- Test with Simple ---
    # print("\nTesting Simple Observation Mode...")
    # env_simple = PacmanEnvironment(render=True, observation_mode='simple')
    # obs_simple = env_simple.reset()
    # print(f"Initial Simple Observation Shape: {obs_simple.shape}")
    # env_simple.current_gen = 9999
    # done_simple = False
    # total_reward_simple = 0.0
    # last_action_simple = None

    key_mappings = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2,
        pygame.K_DOWN: 3
    }

    # Choose which environment to run interactively
    env = env_minimap # Change to env_simple to test simple mode
    env.debug = 1
    env.MAX_EPISODE_STEPS = 99999 # Set a max step limit for the interactive run
    total_reward = total_reward_minimap
    last_action = last_action_minimap
    done = done_minimap
    
    # env.game_state.no_ghosts = [True, False, False, False] # Uncomment to test with only one ghost

    clock = pygame.time.Clock()

    while not done:
        action = last_action # Repeat last action if no key pressed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                mapped = key_mappings.get(event.key)
                if mapped is not None:
                    action = mapped

        if action is not None and not done:
            obs, reward, step_done, info = env.step(action)
            total_reward += reward
            last_action = action
            done = step_done
            if env.debug == 1:
                # print ghosts positions:
                for i, name in enumerate(env.ghost_order):
                    if obs[i*2] != -1 and obs[i*2 + 1] != -1:
                        print(f"{name} position: ({obs[i*2]}, {obs[i*2 + 1]})")
                print(f"ghost scared bits {obs[8:12]}")
                print(f"closest dot {obs[12:14]}")
                print(f"closest power  up {obs[14:16]}")
                print(f"remaining dots {obs[16]}")
                print(f"wall distances {obs[17:21]}")
                print(f"Pacman power state: {obs[21]}")
                print(f"Pacman last action: {obs[22]}")
                print(f"Minimap:\n", obs[26:90].reshape(8, 8))
                
            elif env.debug == 2:
                print(f"Action: {action}, Reward: {reward:.3f}, Total Reward: {total_reward:.3f}, Done: {done}")
                if env.observation_mode == 'minimap':
                    print("Minimap Observation:\n", obs[26:90].reshape(8, 8))
                else:
                    print("Simple Observation:", obs)

        # Limit frame rate
        clock.tick(60) # Aim for 60 FPS

    print(f"Game Over ({env.observation_mode} mode), total reward:", total_reward)
    print(f"Final Score: {env.game_state.points}")

    env.close()