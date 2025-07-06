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

    def __init__(self, render : bool, obs_mode : str = 'minimap'):
        
        #? -------------------------- Game state parameters --------------------------
        self.game_state         : GameState             = None          # GameState object
        self.screen             : pygame.Surface        = None          # Pygame surface
        self.event_handler      : EventHandler          = None          # EventHandler object
        self.all_sprites        : pygame.sprite.Group   = None          # Pygame sprite group
        self.screen_manager     : ScreenManager         = None          # ScreenManager object
        self.render_enabled     : bool                  = render        # Whether to render the game
        self.debug              : int                   = 0             # Debug level
        self.MAX_EPISODE_STEPS  : int                   = 5000          # Default max steps per episode
        
        #? -------------------------- Game state variables --------------------------
        self.ghost_order        : list[str]             = ['blinky', 'pinky', 'inky', 'clyde'] # Ghost order
        self.px                 : int                   = 0             # Pacman's x-coordinate in grid
        self.py                 : int                   = 0             # Pacman's y-coordinate in grid
        self.visits             : np.ndarray            = None          # Array of visits to each cell
        self.last_action        : int                   = None          # Last action taken
        self.pacman_pos         : np.ndarray            = None          # Pacman's position in grid
        self.max_reached        : bool                  = False         # Whether the max steps have been reached
        
        #? -------------------------- Observation variables --------------------------
        self.obs_mode   : str                   = obs_mode
        self.wall_distance_map  : np.ndarray            = None          # Pre-computed wall distances
        self.wall_code          : int                   = None          # Will be set in reset()
        self.dot_code           : int                   = None          # "dot"  tile code
        self.power_code         : int                   = None          # "power" tile code
        self.dot_positions      : np.ndarray            = None          # np.ndarray of dot coordinates (rows, cols)
        self.power_positions    : np.ndarray            = None          # np.ndarray of power-up coordinates
        
        # Pre-compute reciprocals of grid dimensions to replace repeated division
        self._inv_num_rows      : float                 = 1.0 / NUM_ROWS
        self._inv_num_cols      : float                 = 1.0 / NUM_COLS
        
        #? -------------------------- Reward variables --------------------------
        self.n_opposite         : int                   = 0             # Number of opposite moves
        self.stuck_counter      : int                   = 0             # Number of consecutive moves in the same direction
        self.no_dot_steps       : int                   = 0             # Number of steps since last dot
        self.prev_points        : int                   = 0             # Previous points
        self.tot_visited        : int                   = 0             # Total number of visited cells
        
        #? -------------------------- Curriculum learning --------------------------
        self.current_gen        : int = 0                               # This will be set in run() or train()
        self.current_level      : int = 1                               # This will be set in reset()
        self.EASY_GEN           : list[int] = [1000, 1000, 1000, 1000]  # Generations at which each ghost is added
        self.progress           : dict[float, bool] = {}                # Progress checkpoints for curriculum learning

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
            
        #? Curriculum learning:
        # 1) Start with level 1 (only dots, ghosts only in chase mode [gen >= 500])
        # 2) After 1500 generations, switch to level 2 (same as level 1 but ghosts have scatter and chase mode)
        # 3) After 2000 generations, switch to level 3 (dots, powerups, ghosts)
        prev_level = getattr(self, "current_level", None)
        if self.current_gen >= 2000:
            level = 3
        elif self.current_gen >= 1500:
            level = 2
        else:
            level = 1
        if prev_level != level:
            print(f"[Curriculum] \t Switching to level {level}")
        self.current_level = level
        
        #? Game state initialization
        self.game_state = GameState(level)
        self.game_state.sound_enabled = False
        self.event_handler = EventHandler(self.screen, self.game_state)
        self.all_sprites = pygame.sprite.Group()

        self.prev_points = 0
        self.stuck_counter = 0
        self.no_dot_steps = 0
        self.last_action = None
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
        
        # Cache tile codes
        self.wall_code  = self.game_state.tile_encoding.get("wall", 0)
        self.dot_code   = self.game_state.tile_encoding.get("dot", 1)
        self.power_code = self.game_state.tile_encoding.get("power", 2)
        
        # Pre-compute wall distances for the whole level grid
        self._precompute_wall_distances()
        
        # Cache initial dot / power-up positions to avoid np.argwhere in hot path
        grid = self.game_state.level_matrix_np
        self.dot_positions   = np.argwhere(grid == self.dot_code)
        self.power_positions = np.argwhere(grid == self.power_code)
        
        if self.current_gen < min(self.EASY_GEN):
            self.MAX_EPISODE_STEPS = 2000
            
        for i in range(len(self.ghost_order)):
            if self.current_gen < self.EASY_GEN[i]:
                self.game_state.no_ghosts[self.ghost_order[i]] = True
            else:
                self.game_state.no_ghosts[self.ghost_order[i]] = False
        
        # Use cached dot positions for efficiency
        self.total_dots = len(self.dot_positions)
        self.remaining_dots = self.total_dots
        
        self.progress = {
            0.9: False,
            0.75: False,
            0.5: False,
            0.25: False
        }
        
        # Get initial observation based on the configured mode
        return self.get_observation(mode=self.obs_mode)

    #?--------------------------------------------------------------
    #?                        Step function
    #?--------------------------------------------------------------

    def step(self, action : int):
        """
        Advance one frame with the specified action.
        Args:
            action (int): Action chosen by NEAT: 0=left, 1=right, 2=up, 3=down
        Returns:
            observation (list or np.array)
            reward (float)
            done (bool)
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

        #? replication of "GameRun.main()" loop but only for 1 iteration
        # Handle Pygame events (important even in headless for QUIT signals etc.)
        for event in pygame.event.get():
            # Pass manual=False as NEAT controls direction
            self.event_handler.handle_events(event, manual=False)

        # Check time-based game events (ghost mode changes, powerup expiry)
        self.event_handler.check_frame_events()

        # Update game logic (sprite movements, collisions)
        self.all_sprites.update(1)
        
        self.prev_px, self.prev_py = self.px, self.py
        
        # Update Pacman's grid position AFTER movement
        self._update_pacman_grid_position()

        # Refresh cached dot / power-up coordinates after state update
        self._refresh_item_positions()

        # Get the current observation
        obs = self.get_observation(mode=self.obs_mode)

        # Using cached arrays we can compute remaining dots quickly
        prev_remaining_dots = self.remaining_dots
        self.remaining_dots = len(self.dot_positions)
        
        # Determine if done
        done = self.game_state.is_pacman_dead or self.game_state.level_complete
        
        # Update steps since last dot counter
        if prev_remaining_dots - self.remaining_dots > 0:
            self.no_dot_steps = 0
        else:
            self.no_dot_steps += 1
            
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

        return obs, reward, done

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

                # dist_left
                dist = 0
                for i in range(c - 1, -1, -1):
                    if grid[r, i] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 0] = dist / cols

                # dist_right
                dist = 0
                for i in range(c + 2, cols):
                    if grid[r, i] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 1] = dist / cols

                # dist_up
                dist = 0
                for i in range(r - 1, -1, -1):
                    if grid[i, c] == wall_code: break
                    dist += 1
                self.wall_distance_map[r, c, 2] = dist / rows

                # dist_down
                dist = 0
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
            
            # Convert pixel coordinates to grid indices
            grid_y, grid_x = get_idx_from_coords(px_pixel, py_pixel, start_x, start_y, CELL_SIZE[0])
            self.py_float, self.px_float = get_float_idx_from_coords(px_pixel, py_pixel, start_x, start_y, CELL_SIZE[0])
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

    def get_observation(self, mode : str = "minimap", minimap_size : int = 8):
        """
        Returns an observation representation of the current game state.

        Args:
            mode (str): Observation type to return. Currently supported: "simple", "minimap".
                        Defaults to "simple".

        Returns:
            if mode == "simple":
                numpy.ndarray: Observation vector (total 26 elements) containing:
                    - Relative normalized positions of up to 4 ghosts (fixed order: 'blinky', 'pinky', 'inky', 'clyde') [8 elements]
                    - Ghost scared bits (1.0 if scared, 0.0 otherwise) [4 elements]
                    - Vector (normalized) from Pacman to the nearest dot [2 elements]
                    - Vector (normalized) from Pacman to the nearest powerup [2 elements]
                    - Number of remaining dots [1 element]
                    - Normalized distances to walls in 4 directions (left, right, up, down) [4 elements]
                    - Pacman's power-up state (1.0 if active, 0.0 otherwise) [1 element]
                    - One-hot encoding of last action [4 elements]
            if mode == "minimap":
                numpy.ndarray: Observation vector (total 26 elements + 64 minimap elements = 90 elements) containing:
                    - All elements from "simple" mode [26 elements]
                    - Minimap representation of the game state [64 elements]
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

            #? (1) Relative positions of ghosts (GRID BASED)
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

            #? (2) Ghost scared bits
            ghost_scared_bits = np.array([1.0 if game_state.scared_ghosts[game_state.ghost_encoding[name]] else 0.0 for name in self.ghost_order], dtype=np.float32)

            #? (3) Compute the vector to the nearest dot
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

            #? (4) Nearest powerup (if any)
            powerups = self.power_positions
            if powerups.size > 0:
                d_sq = np.sum((powerups - self.pacman_pos) ** 2, axis=1)
                idx = np.argmin(d_sq)
                nearest_powerup_grid = powerups[idx]
                diff_p = nearest_powerup_grid - self.pacman_pos
                closest_powerup_vec = np.array([diff_p[0] * self._inv_num_rows, diff_p[1] * self._inv_num_cols])
            else:
                closest_powerup_vec = np.array([-1, -1])
                                
            #? (5) Remaining dots
            remaining_dots = [self.remaining_dots/self.total_dots]

            #? (6) Get pre-computed distances to the nearest wall in 4 directions
            r, c = pacman_grid.astype(int)
            wall_dists = self.wall_distance_map[r, c]

            #? (7) Pacman powerup status
            power_state = np.array([game_state.is_pacman_powered])
            
            #? (8) One hot encoding of last action
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
            #? (9) Minimap
            minimap_size = 8
            radius = minimap_size // 2

            # Define encoding for different elements in the minimap
            encoding = {
                "wall"          : -1.0,     # Walls
                "elec"          : -1.0,     # Treat electric fence like a wall
                "out_of_bounds" : -1.0,     # Treat out of bounds like a wall
                "ghost_normal"  : -0.75,    # Ghosts
                "void"          : 0.0,      # Empty space Pacman can move into
                "dot"           : 0.5,      # Dots
                "power"         : 0.75,     # Power pellets
                "ghost_scared"  : 1.0,      # Scared ghosts
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

                        # Set the minimap cell value (ghosts overwrite static elements)
                        minimap[r_mini, c_mini] = encoding["ghost_scared"] if game_state.scared_ghosts[ghost_name_to_idx[name]] else encoding["ghost_normal"]

            # Return the flattened minimap as a 1D vector concatenated with the observation
            return np.concatenate([observation, minimap.flatten()])

        else:
            print(f"Warning: Unknown observation mode '{mode}'. Returning empty array.")
            return np.array([])

    #?-------------------------------------------------------------------------
    #?                          Complex Reward function
    #?-------------------------------------------------------------------------

    def _calculate_reward(self, previous_points, current_observation):
        """
        Calculates the reward for the current step.
        Args:
            previous_points (int): Score before the current step.
            current_observation (np.array or None): The observation vector
                from simple mode IF self.obs_mode is 'simple'. Otherwise None.
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
        
        #? -------------------------- Cost of Living --------------------------
        # COST_OF_LIVING = -0 #?
        # reward += COST_OF_LIVING
        # if self.debug >= 3: print(f"cost of living: {COST_OF_LIVING}")

        #? -------------------------- Reward for Points Gained --------------------------
        # Dynamically scale rewards for points gained: as more dots are eaten,
        # the multiplier increases, making each dot or power-up more valuable near the end of the level.
        SCORE_MULTIPLIER = 1 + 9 * ((self.total_dots - self.remaining_dots) / self.total_dots)
        DOT_EATEN_BONUS = 10.0
        POWER_PELLET_EATEN_BONUS = 20.0
        GHOST_EATEN_BONUS = 150.0
        
        points_gain = current_points - previous_points
        last_reward = reward
        tmp = ""

        if points_gain == 10: # Pacman ate a standard dot
            reward += 10 * SCORE_MULTIPLIER
            reward += DOT_EATEN_BONUS
            tmp = "dot"
        elif points_gain == 15: # Pacman ate a power pellet
            reward += 50 * SCORE_MULTIPLIER
            reward += POWER_PELLET_EATEN_BONUS
            tmp = "power up"
        elif points_gain == 25: # Pacman ate a ghost
            reward += GHOST_EATEN_BONUS
            tmp = "ghost"
        elif points_gain > 0:
            reward += points_gain * SCORE_MULTIPLIER
            tmp = "other"
        if self.debug >= 3 and points_gain > 0: print(f"Pacman ate a {tmp}, reward: {reward - last_reward}")

        #? -------------------------- Penalty for Death --------------------------
        DEATH_PENALTY = -1000
        if game_state.is_pacman_dead:
            reward += DEATH_PENALTY
            if self.debug >= 3: print(f"Pacman died, reward: {DEATH_PENALTY}")

        #? -------------------------- Bonus for Level Completion --------------------------
        LEVEL_COMPLETE_BONUS = 10000
        if game_state.level_complete:
            reward += LEVEL_COMPLETE_BONUS
            if self.debug >= 3: print(f"Pacman completed the level, reward: {LEVEL_COMPLETE_BONUS}")

        #? -------------------------- Penalty for Getting Stuck Against Wall --------------------------
        if current_observation is not None:
            STUCK_PENALTY_FACTOR = -0.1
            STUCK_PENALTY_CAP = -0.5 # Limit penalty increase
            STUCK_LIMIT = 100 # Limit for stuck penalty
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
                
        #? -------------------------- Reward/Penalty for Ghost Interaction --------------------------
        GHOST_INTERACTION_FACTOR = 0.5
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
                        gh_points = sign * GHOST_INTERACTION_FACTOR * closeness_factor
                        gh_reward += gh_points
                        
                        if self.debug >= 3: print(f"Pacman close to ghost {name} ({"scared" if is_scared else "not scared"}), distance: {dist_grid_sq}, reward: {gh_points}")

            reward += gh_reward

        #? -------------------------- Penalty for Immediate Reversal (Anti-Oscillation) --------------------------
        # REVERSAL_PENALTY = -0.15
        # MAX_REVERSAL_PENALTY = 3 # Maximum penalty for reversal
        # last_reward = reward
        # if self.last_action is not None and current_action_int is not None:
        #     # Check if actions are opposite (l<->r or u<->d)
        #     is_opposite = abs(current_action_int - self.last_action) == 1 and current_action_int // 2 == self.last_action // 2
        #     # Apply penalty only if no points were gained (didn't just eat something)
        #     if is_opposite and points_gain <= 0:
        #         self.n_opposite += 1
        #         reward += REVERSAL_PENALTY*min(self.n_opposite, MAX_REVERSAL_PENALTY)
        #         if self.debug >= 3: print(f"Pacman reversed direction, reward: {reward - last_reward}")
        #     else:
        #         self.n_opposite = 0
                
        #? -------------------------- Exploration / Visit Penalty --------------------------
        VISIT_BONUS_FIRST =          1 # Bonus for first visit
        VISIT_PENALTY_THRESHOLD =    4 # Start penalizing later
        VISIT_PENALTY_FACTOR =    -0.1 # Penalty factor for each visit over threshold
        VISIT_PENALTY_CAP =         -1 # Cap penalty to avoid excessive negative rewards

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
                 
        #? -------------------------- Penalty for not eating dots --------------------------
        NO_DOT_PENALTY_START_STEP =  100 # Start penalizing after 50 steps without eating
        NO_DOT_PENALTY_FACTOR =    -0.05 # Penalty per step after threshold
        NO_DOT_PENALTY_CAP =          -1 # Maximum penalty per step for this

        last_reward = reward
        if self.no_dot_steps > NO_DOT_PENALTY_START_STEP:
            no_dot_penalty = NO_DOT_PENALTY_FACTOR * (self.no_dot_steps - NO_DOT_PENALTY_START_STEP)
            reward += max(NO_DOT_PENALTY_CAP, no_dot_penalty)
            if self.debug >= 3: print(f"Pacman hasn't eaten a dot in {self.no_dot_steps} steps, reward: {reward - last_reward}")
      
        if self.debug >= 3: print(f"Total reward: {reward:.3f}\n")
        
        #? -------------------------- One-time Checkpoint Bonuses --------------------------
        # This is added to the reward function below to avoid code repetition
        
        return reward
    
    #?--------------------------------------------------------------------------------------
    #?                      Reward only based on survival time (steps)
    #?--------------------------------------------------------------------------------------
    def calculate_reward(self, previous_points, current_observation):
        """
        This is the reward function used in the first training phase.
        It is a simple reward function that only rewards Pacman for surviving and eating dots.
        It is used to train the agent to learn the basic mechanics of the game.
        """
        bonus = 250
        if self.current_gen > 2000:
            reward = self._calculate_reward(previous_points, current_observation)
            bonus = 500
        else:
            reward = self.game_state.points - previous_points 
            reward -= 0.1 # cost of living
        
        eaten_dots = self.total_dots - self.remaining_dots
        eaten_ratio = eaten_dots / self.total_dots if self.total_dots > 0 else 0

        #? -------------------------- One-time Checkpoint Bonuses --------------------------
        if eaten_ratio > 0.9 and not self.progress[0.9]:
            reward += 2*bonus
            self.progress[0.9] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 90%! +{2*bonus}")
            
        elif eaten_ratio > 0.75 and not self.progress[0.75]:
            reward += bonus
            self.progress[0.75] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 75%! +{bonus}")
            
        elif eaten_ratio > 0.5 and not self.progress[0.5]:
            reward += bonus
            self.progress[0.5] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 50%! +{bonus}")
        elif eaten_ratio > 0.25 and not self.progress[0.25]:
            reward += bonus
            self.progress[0.25] = True
            if self.debug >= 2: print(f"CHECKPOINT BONUS: Reached 25%! +{bonus}")
        
        return reward
    
    #?--------------------------------------------------------------
    #?                      Close the environment
    #?--------------------------------------------------------------

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
            
            
            
            
#!-----------------------------------------------------------------------------
#!                              Interactive testing
#!-----------------------------------------------------------------------------

if __name__ == "__main__":
    import pygame

    print("Testing Minimap Observation Mode...")
    env_minimap = PacmanEnvironment(render=True, obs_mode='minimap')
    env_minimap.current_gen = 9999
    obs_minimap = env_minimap.reset()
    print(f"Initial Minimap Observation Shape: {obs_minimap.shape}")

    key_mappings = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2,
        pygame.K_DOWN: 3
    }

    # Choose which environment to run interactively
    env = env_minimap
    env.debug = 1
    total_reward = 0
    last_action = None
    done = False
    
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
            obs, reward, step_done = env.step(action)
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
                if env.obs_mode == 'minimap':
                    print("Minimap Observation:\n", obs[26:90].reshape(8, 8))
                else:
                    print("Simple Observation:", obs)

        # Limit frame rate
        clock.tick(60) # Aim for 60 FPS

    print(f"Game Over ({env.obs_mode} mode), total reward:", total_reward)
    print(f"Final Score: {env.game_state.points}")

    env.close()