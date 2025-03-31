import os
import pygame
import time
import sys
import numpy as np

pacman_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pacman"))
if pacman_path not in sys.path:
    sys.path.insert(0, pacman_path)
    
from src.game.state_management import GameState
from src.game.event_management import EventHandler
from src.gui.screen_management import ScreenManager
from src.utils.coord_utils import get_idx_from_coords
from src.configs import SCREEN_WIDTH, SCREEN_HEIGHT, CELL_SIZE, NUM_ROWS, NUM_COLS

#?--------------------------------------------------------------
#?                    Environment class
#?--------------------------------------------------------------

class PacmanEnvironment:
    """
    Gym-like environment for Pacman.
    Handles initialization, stepping with an action,
    returning observations and rewards, etc.
    """

    def __init__(self, render):
        """
        Args:
            config (dict): Configuration dictionary loaded from YAML.
                           Should contain game settings (e.g. headless flag).
        """
        self.game_state = None
        self.screen = None
        self.event_handler = None
        self.all_sprites = None
        self.screen_manager = None
        self.prev_points = 0
        #self.prev_pos = None
        #self.pos = None
        self.stuck_counter = 0
        self.prev_dist_to_closest_dot = float('inf')
        self.prev_dist_to_closest_powerup = float('inf')


        # Store observation/action dimensions for your NEAT usage
        self.render_enabled = render
        
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
            pygame.init()
            # Create a proper dummy surface with hardware accel disabled
            pygame.display.set_mode((1024, 768), pygame.NOFRAME, 32)
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((1024, 768))
            
        self.game_state = GameState()
        self.game_state.sound_enabled = False
        
        # Create the screen surface based on render mode
        if not self.render_enabled:
            self.screen = pygame.Surface((1024, 768))
        
        # Initialize remaining components
        self.event_handler = EventHandler(self.screen, self.game_state)
        self.all_sprites = pygame.sprite.Group()
        
        # Add tracking for reward calculation
        self.prev_points = 0
        
        try:
            self.screen_manager = ScreenManager(self.screen, self.game_state, self.all_sprites)
        except pygame.error as e:
            print(f"Error initializing screen manager: {e}")
            # Try a fallback approach if there was an error
            
        self.all_sprites.update(0)
            
        obs = self.get_observation()
        
        #self.prev_pos = self.game_state.pacman_rect[:2]
        #self.pos = self.game_state.pacman_rect[:2]
        
        self.last_action_stuck = False 
        
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
        #    For instance:
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
            #self.game_state.direction = ""

        # 2) Update one frame
        reward = 0.0
        done = False

        #? replication of "GameRun.main()" loop but only for 1 iteration

        for event in pygame.event.get():
            self.event_handler.handle_events(event, manual=False)
        
        self.event_handler.check_frame_events()

        # If rendering, fill the screen, draw, etc.
        if self.render_enabled:
            self.screen.fill((0,0,0))
            self.screen_manager.draw_screens()
            self.all_sprites.draw(self.screen)

        # Actually update game logic
        dt = 1
        self.all_sprites.update(dt)

        # Determine if done
        if self.game_state.is_pacman_dead or self.game_state.level_complete:
            done = True

        obs = self.get_observation() # Osservazione *dopo* lo step
        done = self.game_state.is_pacman_dead or self.game_state.level_complete
        reward = self.calculate_reward(current_points, obs, done)
        
        self.prev_points = self.game_state.points
        
        # Construct next observation
        # obs = self.get_observation()

        # Info dict can hold debugging info if desired
        info = {}

        # If rendering, flip the display
        if self.render_enabled:
            pygame.display.flip()
            
        # Update the step count
        self.game_state._step_count += 1

        return obs, reward, done, info

    def get_observation(self, mode = "simple"):
        """
        Returns an observation representation of the current game state.
        
        Args:
            mode (str): Observation type to return. Currently supported: "simple".
                Defaults to "simple".
                
        Returns:
            numpy.ndarray: Observation vector containing:
            1. Pacman’s normalized screen position (x, y)
            2. Normalized positions for up to 4 ghosts (in fixed order)
            3. Vector to the nearest dot/pellet (normalized)
            4. Distances to nearest walls in 4 directions (normalized)
            5. Power-up state (1 if active, 0 otherwise)
        """
        if mode == "simple":
            # (1) Convert Pacman's pixel position into grid coordinates.
            start_x, start_y = self.game_state.start_pos  # Ensure this is set by your level loading
            cell_size = CELL_SIZE[0]
            
            px, py, pxx, pyy = self.game_state.pacman_rect
            pacman_grid = np.array(get_idx_from_coords(px, py, start_x, start_y, cell_size))
            pacman_grid_end = np.array(get_idx_from_coords(pxx, pyy, start_x, start_y, cell_size))

            # Normalize Pacman's position in [0,1] by (NUM_ROWS, NUM_COLS)
            pacman_pos = pacman_grid / np.array([NUM_ROWS, NUM_COLS])
            
            #self.prev_pos = self.pos
            #self.pos = px, py

            # (2) Convert ghost positions (in pixels) into grid coordinates and compute the relative difference.
            ghost_order = ['blinky', 'pinky', 'inky', 'clyde']
            ghost_rel = np.zeros(8)  # 4 ghosts * 2 coords

            ghosts = self.game_state.ghosts  # dict: ghost_name -> (gx, gy) in pixels
            for i, name in enumerate(ghost_order):
                if name in ghosts:
                    gx, gy = ghosts[name]
                    # delta_r, delta_c
                    rel_r = (gx - pacman_grid[0]) / NUM_ROWS
                    rel_c = (gy - pacman_grid[1]) / NUM_COLS
                    ghost_rel[i*2]     = rel_r
                    ghost_rel[i*2 + 1] = rel_c
                else:
                    # If the ghost doesn't exist or isn't placed, we leave (0,0)
                    pass
                
            # (3) Compute the vector to the nearest dot.
            grid = self.game_state.level_matrix_np  # 2D array: 0=wall,1=dot,2=power...
            dot_code = self.game_state.tile_encoding.get("dot", 1)

            # Find all cells containing a dot
            dots = np.argwhere(grid == dot_code)
            if dots.size > 0:
                # Euclidean distance to each dot
                distances = np.linalg.norm(dots - pacman_grid, axis=1)
                idx = np.argmin(distances)
                nearest_dot = dots[idx]
                # Delta normalized by (NUM_ROWS, NUM_COLS)
                closest_dot = (nearest_dot - pacman_grid) / np.array([NUM_ROWS, NUM_COLS])
            else:
                # No dots remain
                closest_dot = np.array([0.0, 0.0])
                
            # (4) Nearest powerup (if any)
            powerup_code = self.game_state.tile_encoding.get("power", 2)
            powerup = np.argwhere(grid == powerup_code)
            if powerup.size > 0:
                distances = np.linalg.norm(powerup - pacman_grid, axis=1)
                idx = np.argmin(distances)
                nearest_powerup = powerup[idx]
                # Delta normalized by (NUM_ROWS, NUM_COLS)
                closest_powerup = (nearest_powerup - pacman_grid) / np.array([NUM_ROWS, NUM_COLS])
            else:
                # No powerup
                closest_powerup = np.array([0.0, 0.0])
                # Normalize the distance to the powerup
                dist_powerup = np.linalg.norm(closest_powerup)
                if dist_powerup > 0:
                    closest_powerup /= dist_powerup
                else:
                    closest_powerup = np.array([0.0, 0.0])
                    
            # (5) Remaining dots
            remaining_dots = [dots.size]

            # (5) Compute distances to the nearest wall in 4 directions (left, right, up, down)
            wall_code = self.game_state.tile_encoding.get("wall", 0)
            r, c = pacman_grid.astype(int)  # integer row/col for scanning
            r_end, c_end = pacman_grid_end.astype(int)  # integer row/col for scanning
            x_offset = abs(r_end - r)
            y_offset = abs(c_end - c)
            max_rows, max_cols = grid.shape

            def distance_to_wall(rr, cc, direction):
                dist = 0
                if direction == "left":
                    for col in range(cc - 1, -1, -1):
                        if grid[rr, col] == wall_code:
                            break
                        dist += 1
                    return dist / cc if cc > 0 else 0.0
                elif direction == "right":
                    for col in range(cc + 2, max_cols):
                        if grid[rr, col] == wall_code:
                            break
                        dist += 1
                    return dist / (max_cols - cc - 1) if (max_cols - cc - 1) > 0 else 0.0
                elif direction == "up":
                    for row in range(rr - 1, -1, -1):
                        if grid[row, cc] == wall_code:
                            break
                        dist += 1
                    return dist / rr if rr > 0 else 0.0
                elif direction == "down":
                    for row in range(rr + 2, max_rows):
                        if grid[row, cc] == wall_code:
                            break
                        dist += 1
                    return dist / (max_rows - rr - 1) if (max_rows - rr - 1) > 0 else 0.0
                return 0.0

            dist_left  = distance_to_wall(r, c, "left")
            dist_right = distance_to_wall(r, c, "right")
            dist_up    = distance_to_wall(r, c, "up")
            dist_down  = distance_to_wall(r, c, "down")
            wall_dists = np.array([dist_left, dist_right, dist_up, dist_down])
            
            # (6) Pacman powerup status
            power_state = np.array([1.0]) if self.game_state.is_pacman_powered else np.array([0.0])

            # Combine everything into one observation vector
            observation = np.concatenate([
                pacman_pos,      # 2
                ghost_rel,       # 8
                closest_dot,     # 2
                closest_powerup, # 2
                remaining_dots,  # 1
                wall_dists,      # 4
                power_state      # 1
            ])                   # tot = 20
            
        return observation
    
    def calculate_reward(self, previous_points, current_observation, done):
        reward = 0.0

        # 1. Reward for Points Gained
        points_gain = self.game_state.points - previous_points
        if points_gain > 0:
            reward += points_gain * 2 # e.g. +10 dot, +15 powerup, +25 ghost
        else:
            reward -= 0.1 # penalty for not eating anything

        # 2. Penalty for Pacman Death
        if self.game_state.is_pacman_dead:
            # penalty proportional to remaining dots (higher penalty if less dots left)
            reward -= 700.0 - current_observation[10]

        # 3. Bonus for Level Completion
        if self.game_state.level_complete:
            reward += 500.0
            
        # map direction to wall check
        dir_decode = { "l": 0, "r": 1, "u": 2, "d": 3 }
         
        # get wall distances from observation
        wall_dists = current_observation[15:19]
        wall_dist = wall_dists[dir_decode[self.game_state.direction]]
        if wall_dist < 1e-3:
            self.stuck_counter += 1
            # If the wall is too close, penalize
            reward -= 0.1 * self.stuck_counter
        else:
            self.stuck_counter = 0

        # 5. Reward for Moving Closer to Dots
        current_dot_vector = current_observation[10:12]
        current_dist_to_dot = np.linalg.norm(current_dot_vector)
        if current_dist_to_dot < self.prev_dist_to_closest_dot and not self.last_action_stuck:
            reward += 0.5
        self.prev_dist_to_closest_dot = current_dist_to_dot
        
        #! Reward getting close to powerups (?)
        # current_powerup_vector = current_observation[12:14]
        # current_dist_to_powerup = np.linalg.norm(current_powerup_vector)
        # if current_dist_to_powerup < self.prev_dist_to_closest_powerup and not self.last_action_stuck:
        #     reward += 1
        # self.prev_dist_to_closest_powerup = current_dist_to_powerup

        # 6. Penalty for getting too close to ghosts
        ghost_relative_positions = current_observation[2:10] # 4 ghosts * 2 coords
        #is_powered_up = current_observation[-1] > 0.5
        if not current_observation[-1]:
            min_dist_to_ghost = float('inf')
            for i in range(0, 8, 2):
                dist = np.linalg.norm(ghost_relative_positions[i:i+2])
                min_dist_to_ghost = min(min_dist_to_ghost, dist)
            if min_dist_to_ghost < 0.1: # Too close to a ghost
                reward -= 2.0

        return reward

    def close(self):
        """
        Shut down the environment, e.g. call pygame.quit() if needed.
        """
        pygame.quit()

if __name__ == "__main__":
    render = True

    # Create the environment
    env = PacmanEnvironment(render)
    # Initialize the environment
    obs = env.reset()
    print("Initial observation:")
    print(obs)
        
    MANUAL = True
        
    if not MANUAL:
        # Run a few steps
        num_steps = 5000
        for step in range(num_steps):
            action = np.random.randint(4)
            obs, reward, done, info = env.step(action)
            print(obs)
            if done:
                print("Episode finished at step", step)
                break

        env.close()
        pygame.quit()
        print("Test completed")
        
    else:
        running = True
        action = None
        
        total_reward = 0.0

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 0  # sinistra
                    elif event.key == pygame.K_RIGHT:
                        action = 1  # destra
                    elif event.key == pygame.K_UP:
                        action = 2  # su
                    elif event.key == pygame.K_DOWN:
                        action = 3  # giù

            if action is None:
                continue

            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            #! debug
            # print(obs)
            print(total_reward)

            if done:
                print("Episode finished")
                running = False
                obs = env.reset()
                action = None  # Resetta l'azione
                
            time.sleep(1/60)  # Delay to control the speed of the game

        env.close()
        pygame.quit()
