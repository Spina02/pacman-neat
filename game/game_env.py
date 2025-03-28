import os
import pygame
import time
import sys
import numpy as np

pacman_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pacman"))
print("Pacman path:", pacman_path)
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

    def __init__(self, config):
        """
        Args:
            config (dict): Configuration dictionary loaded from YAML.
                           Should contain game settings (e.g. headless flag).
        """
        self.config = config
        self.game_state = None
        self.screen = None
        self.event_handler = None
        self.all_sprites = None
        self.screen_manager = None

        # Store observation/action dimensions for your NEAT usage
        self.observation_size = config["game"]["observation_size"]
        self.action_size = config["game"]["action_size"]
        self.render_enabled = config["game"]["render"]
        
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
        self.game_state.mode_change_events = [7, 7, 5, 20]
        
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
            
        return self.get_observation()
    
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
            self.game_state.direction = ""

        # 2) Update one frame
        reward = 0.0
        done = False

        #? replication of "GameRun.main()" loop but only for 1 iteration

        for event in pygame.event.get():
            self.event_handler.handle_events(event)

        # If rendering, fill the screen, draw, etc.
        if self.render_enabled:
            self.screen.fill((0,0,0))
            self.screen_manager.draw_screens()
            self.all_sprites.draw(self.screen)

        # Actually update game logic
        dt = 1
        self.all_sprites.update(dt)

        # Gather the immediate reward
        reward = self.calculate_reward()

        # Determine if done
        if self.game_state.is_pacman_dead or self.game_state.level_complete:
            done = True

        # Construct next observation
        obs = self.get_observation()

        # Info dict can hold debugging info if desired
        info = {}

        # If rendering, flip the display
        if self.render_enabled:
            pygame.display.flip()
            
        time.sleep(0.01)

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

            # px, py = self.game_state.pacman_rect[:2]  # Pacman's top-left in pixels
            # pacman_grid = np.array(
            #     get_idx_from_coords(px, py, start_x, start_y, cell_size)
            # )  # shape: (2,) => [row, col]
            
            px, py, pxx, pyy = self.game_state.pacman_rect
            pacman_grid = np.array(get_idx_from_coords(px, py, start_x, start_y, cell_size))
            pacman_grid_end = np.array(get_idx_from_coords(pxx, pyy, start_x, start_y, cell_size))

            # Normalize Pacman's position in [0,1] by (NUM_ROWS, NUM_COLS)
            pacman_pos = pacman_grid / np.array([NUM_ROWS, NUM_COLS])

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
                wall_dists,      # 4
                power_state      # 1
            ])                   # tot = 19
            
        return observation
              
    def calculate_reward(self):
        """
        Return a scalar reward each step. 
        Common approach: difference in `game_state.points` 
        from previous frame to current frame.
        """
        # You can store self.prev_points and do reward = current_points - prev_points
        # Then update self.prev_points = current_points, etc.
        return float(self.game_state.points)

    def close(self):
        """
        Shut down the environment, e.g. call pygame.quit() if needed.
        """
        pygame.quit()

if __name__ == "__main__":
    config = { "game": { "observation_size": 10,
                        "action_size": 4,
                        "render": True
                        } }

    # Create the environment
    env = PacmanEnvironment(config)
    obs = env.reset()
    print("Initial observation:")
    for row in obs:
        print(row)
        
    MANUAL = True
        
    if not MANUAL:
        # Run a few steps
        num_steps = 5000
        for step in range(num_steps):
            # random action in config["game"]["action_size"]
            action = np.random.randint(config["game"]["action_size"])
            obs, reward, done, info = env.step(action)
            #print(f"Step {step}: reward = {reward}, done = {done}")
            for row in obs:
                print(row)
            if done:
                print("Episode finished at step", step)
                break
            # Slow down the loop [Optional]
            time.sleep(0.01)

        env.close()
        pygame.quit()
        print("Test completed")
    else:
        running = True
        action = None

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
            
            #! debug
            print(obs)

            if done:
                print("Episode finished")
                obs = env.reset()
                action = None  # Resetta l'azione

            time.sleep(0.01)

        env.close()
        pygame.quit()
