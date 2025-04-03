import os
import pygame
import sys
import numpy as np

pacman_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pacman"))
if pacman_path not in sys.path:
    sys.path.insert(0, pacman_path)
    
from src.game.state_management import GameState
from src.game.event_management import EventHandler
from src.gui.screen_management import ScreenManager
from src.utils.coord_utils import get_idx_from_coords, pixel_to_grid
from src.configs import CELL_SIZE, NUM_ROWS, NUM_COLS, TILE_SIZE

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
        self.game_state = None
        self.screen = None
        self.event_handler = None
        self.all_sprites = None
        self.screen_manager = None
        self.prev_points = 0
        self.stuck_counter = 0
        self.prev_dist_to_closest_dot = float('inf')
        self.render_enabled = render
        self.px = 0
        self.py = 0
        self.visits = None
        self.neg_count = 0
        self.current_gen = 0
        self.steps_since_last_dot = 0
        self.last_action = None
        
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
            pygame.display.set_mode((1024, 768), pygame.NOFRAME, 32)
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((1024, 768))
            
        self.game_state = GameState()
        self.game_state.sound_enabled = False
        
        if not self.render_enabled:
            self.screen = pygame.Surface((1024, 768))
        
        self.event_handler = EventHandler(self.screen, self.game_state)
        self.all_sprites = pygame.sprite.Group()
        
        self.prev_points = 0
        self.neg_count = 0
        self.prev_dist_to_closest_dot = float('inf')
        self.stuck_counter = 0
        self.steps_since_last_dot = 0
        self.last_action = None
        
        self.screen_manager = ScreenManager(self.screen, self.game_state, self.all_sprites)
            
        self.all_sprites.update(0)
        
        rows, cols = self.game_state.level_matrix_np.shape
        self.visits = np.zeros((rows, cols), dtype=int)
        
        START_MAX_STEPS = 1500
        FINAL_MAX_STEPS = 4000
        GENS_TO_REACH_FINAL = 200 # Reach final limit after 150 generations
        
        if self.current_gen < GENS_TO_REACH_FINAL:
            self.MAX_EPISODE_STEPS = int(START_MAX_STEPS + (FINAL_MAX_STEPS - START_MAX_STEPS) * (self.current_gen / GENS_TO_REACH_FINAL))
        else:
            self.MAX_EPISODE_STEPS = FINAL_MAX_STEPS
        
        obs = self.get_observation()
        
        self.remaining_dots = obs[18]
        
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

        # 2) Update one frame
        reward = 0.0
        done = False

        #? replication of "GameRun.main()" loop but only for 1 iteration

        for event in pygame.event.get():
            self.event_handler.handle_events(event, manual=False)
        
        self.event_handler.check_frame_events()

        if self.render_enabled:
            self.screen.fill((0, 0, 0))
            self.screen_manager.draw_screens()
            self.all_sprites.draw(self.screen)

        # Actually update game logic
        dt = 1
        self.all_sprites.update(dt)
        
        previous_dots = self.remaining_dots

        # Get the current observation
        obs = self.get_observation()
        
        self.remaining_dots = obs[18]
        
        # Determine if done
        done = self.game_state.is_pacman_dead or self.game_state.level_complete
        
        # Update steps since last dot counter
        dots_eaten_this_step = previous_dots - self.remaining_dots
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
            done = True

        self.prev_points = self.game_state.points

        info = {}
        if self.render_enabled:
            pygame.display.flip()
            
        self.game_state.step_count += 1
        return obs, reward, done, info
    
    #?--------------------------------------------------------------
    #?                    Observation function
    #?--------------------------------------------------------------

    def get_observation(self, mode="simple"):
        """
        Returns an observation representation of the current game state.

        Args:
            mode (str): Observation type to return. Currently supported: "simple".
                        Defaults to "simple".

        Returns:
            numpy.ndarray: Observation vector (total 20 elements) containing:
                - Pacman's normalized grid position (row, col) [2 elements]
                - Relative normalized positions of up to 4 ghosts (fixed order: 'blinky', 'pinky', 'inky', 'clyde') [8 elements]
                - Vector (normalized) from Pacman to the nearest dot [2 elements]
                - Vector (normalized) from Pacman to the nearest powerup [2 elements]
                - Number of remaining dots [1 element]
                - Normalized distances to walls in 4 directions (left, right, up, down) [4 elements]
                - Pacman's power-up state (1.0 if active, 0.0 otherwise) [1 element]
        """
        if mode == "simple":
            start_x, start_y = self.game_state.start_pos
            cell_size = CELL_SIZE[0]
            
            self.px, self.py = self.game_state.pacman_rect[:2]
            pacman_grid = np.array(get_idx_from_coords(self.px, self.py, start_x, start_y, cell_size))
            self.px = int((self.px - start_x) / cell_size)
            self.py = int((self.py - start_y) / cell_size)
            pacman_pos = pacman_grid / np.array([NUM_ROWS, NUM_COLS])
            
            # (2) Relative positions of ghosts
            ghost_order = ['blinky', 'pinky', 'inky', 'clyde']
            ghost_rel = np.zeros(8)  # 4 ghosts * 2 coords
            ghosts = self.game_state.ghosts  # dict: ghost_name -> (gx, gy) in pixels
            for i, name in enumerate(ghost_order):
                if name in ghosts:
                    gx, gy = ghosts[name]
                    rel_r = (gx - pacman_grid[0]) / NUM_ROWS
                    rel_c = (gy - pacman_grid[1]) / NUM_COLS
                    ghost_rel[i*2]     = rel_r
                    ghost_rel[i*2 + 1] = rel_c
                else:
                    ghost_rel[i*2]     = -1
                    ghost_rel[i*2 + 1] = -1
                # If the ghost doesn't exist or isn't placed, leave (1,1)

            # (3) Ghost scared bits
            ghost_scared_bits = np.array(self.game_state.scared_ghosts, dtype=int)
                
            # (4) Compute the vector to the nearest dot.
            grid = self.game_state.level_matrix_np  # 2D array: 0=wall,1=dot,2=power...
            dot_code = self.game_state.tile_encoding.get("dot", 1)

            # Find all cells containing a dot
            dots = np.argwhere(grid == dot_code)
            if dots.size > 0:
                distances = np.linalg.norm(dots - pacman_grid, axis=1)
                idx = np.argmin(distances)
                nearest_dot = dots[idx]
                closest_dot = (nearest_dot - pacman_grid) / np.array([NUM_ROWS, NUM_COLS])
            else:
                closest_dot = np.array([0.0, 0.0])
                
            # (5) Nearest powerup (if any)
            powerup_code = self.game_state.tile_encoding.get("power", 2)
            powerup = np.argwhere(grid == powerup_code)
            if powerup.size > 0:
                distances = np.linalg.norm(powerup - pacman_grid, axis=1)
                idx = np.argmin(distances)
                nearest_powerup = powerup[idx]
                closest_powerup = (nearest_powerup - pacman_grid) / np.array([NUM_ROWS, NUM_COLS])
            else:
                closest_powerup = np.array([0.0, 0.0])
                dist_powerup = np.linalg.norm(closest_powerup)
                if dist_powerup > 0:
                    closest_powerup /= dist_powerup
                else:
                    closest_powerup = np.array([0.0, 0.0])
                    
            # (6) Remaining dots
            remaining_dots = [dots.size]

            # (7) Compute distances to the nearest wall in 4 directions (left, right, up, down)
            wall_code = self.game_state.tile_encoding.get("wall", 0)
            r, c = pacman_grid.astype(int)
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
            
            # (8) Pacman powerup status
            power_state = np.array([1.0]) if self.game_state.is_pacman_powered else np.array([0.0])

            # Combine everything into one observation vector
            observation = np.concatenate([
                pacman_pos,        # 2
                ghost_rel,         # 8
                ghost_scared_bits, # 4
                closest_dot,       # 2
                closest_powerup,   # 2
                remaining_dots,    # 1
                wall_dists,        # 4
                power_state        # 1
            ])                     # Total: 24 elements
            return observation
        else:
            # TODO: Implement other observation modes
            return np.array([])
        
    #?--------------------------------------------------------------
    #?                      Reward function
    #?--------------------------------------------------------------

    def calculate_reward(self, previous_points, current_observation):
        reward = 0.0
        current_points = self.game_state.points

        # 1. Cost of Living
        COST_OF_LIVING = -0.005
        reward += COST_OF_LIVING

        # 2. Reward for Points Gained
        SCORE_MULTIPLIER = 6
        points_gain = current_points - previous_points

        # 3. Specific Event Bonuses
        DOT_EATEN_BONUS = 8.0
        POWER_PELLET_EATEN_BONUS = 20.0
        GHOST_EATEN_BONUS = 50.0

        if points_gain == 10: # Pacman ate a standard dot
            reward += points_gain * SCORE_MULTIPLIER
            reward += DOT_EATEN_BONUS
        elif points_gain == 15: # Pacman ate a power pellet
            reward += points_gain * SCORE_MULTIPLIER
            reward += POWER_PELLET_EATEN_BONUS
        elif points_gain == 25: # Pacman ate a ghost
            reward += points_gain * SCORE_MULTIPLIER
            reward += GHOST_EATEN_BONUS
        elif points_gain > 0:
            reward += points_gain * SCORE_MULTIPLIER

        # 3. Penalty for Death
        if self.game_state.is_pacman_dead:
            reward -= 500
            
        # 4. Bonus for Level Completion
        if self.game_state.level_complete:
            reward += 15000

        #5. Penalty for Getting Stuck Against Wall (Capped counter)
        STUCK_PENALTY_FACTOR = -0.1
        MAX_STUCK_COUNT_FOR_PENALTY = 10 # Stop increasing penalty after 10 consecutive stuck steps
        dir_decode = {"l": 0, "r": 1, "u": 2, "d": 3}
        wall_dists = current_observation[19:23]
        current_dir_idx = dir_decode.get(self.game_state.direction)

        # is_stuck = False
        if current_dir_idx is not None:
            wall_dist = wall_dists[current_dir_idx]
            if wall_dist < 1e-3: # Check if against wall
                self.stuck_counter += 1
                # is_stuck = True
                stuck_penalty = STUCK_PENALTY_FACTOR * min(self.stuck_counter, MAX_STUCK_COUNT_FOR_PENALTY)
                reward += stuck_penalty
            else:
                self.stuck_counter = 0
        else:
            self.stuck_counter = 0

        # #6. Reward for Moving Closer to Dots (Only if not stuck)
        # APPROACH_DOT_REWARD = 0.35
        # current_dot_vector = current_observation[14:16]
        # # Check if dots remain (vector is not [0,0])
        # if np.any(current_dot_vector):
        #     current_dist_to_dot = np.linalg.norm(current_dot_vector)
        #     if current_dist_to_dot < self.prev_dist_to_closest_dot and not is_stuck:
        #         reward += APPROACH_DOT_REWARD
        #     self.prev_dist_to_closest_dot = current_dist_to_dot
        # else:
        #     # No dots left, maybe set distance to 0 or handle differently
        #     self.prev_dist_to_closest_dot = 0.0

        # 7. Reward/Penalty for Ghost Interaction
        GHOST_INTERACTION_FACTOR = 2.5 # Tune
        if not self.game_state.no_ghosts:
            ghost_relative_positions = current_observation[2:10]
            scared_ghosts = current_observation[10:14]
            gh_reward = 0.0
            for i in range(4):
                # Consider only active ghosts
                ghost_pos = ghost_relative_positions[2*i:2*i+2]
                if np.any(ghost_pos != -1): # Check if ghost exists / is active
                    dist = np.linalg.norm(ghost_pos)
                    if dist > 1e-6:
                        proximity_threshold = 0.2 # Fraction of map size
                        if dist < proximity_threshold:
                            closeness_factor = (proximity_threshold - dist) / proximity_threshold # 0 (at threshold) to 1 (at dist 0)
                            sign = 1.0 if scared_ghosts[i] else -1.0
                            gh_points = sign * GHOST_INTERACTION_FACTOR * closeness_factor**2 # Penalize quadratically closer
                            gh_reward += gh_points
            reward += gh_reward

        # 8. Exploration / Visit Penalty
        VISIT_BONUS_FIRST =         1 # Bonus for first visit
        VISIT_BONUS_SECOND =      0.5 # Bonus for second
        VISIT_PENALTY_THRESHOLD =  12 # Start penalizing later
        VISIT_PENALTY_FACTOR =   -0.1 # Penalty factor for each visit over threshold
        VISIT_PENALTY_CAP =     -0.75 # Cap penalty to avoid excessive negative rewards

        rows, cols = self.visits.shape
        current_row, current_col = int(self.py), int(self.px)
        if 0 <= current_row < rows and 0 <= current_col < cols:
            visit_count = self.visits[current_row, current_col]

            if visit_count == 0:
                reward += VISIT_BONUS_FIRST
            elif visit_count == 1:
                reward += VISIT_BONUS_SECOND
            elif visit_count > VISIT_PENALTY_THRESHOLD:
                visit_penalty = VISIT_PENALTY_FACTOR * (visit_count - VISIT_PENALTY_THRESHOLD)
                reward += max(VISIT_PENALTY_CAP, visit_penalty)
                
        # 9. NEW: Penalty for not eating dots
        NO_DOT_PENALTY_START_STEP = 20 # Start penalizing after 25 steps without eating
        NO_DOT_PENALTY_FACTOR =  -0.05 # Penalty per step after threshold
        NO_DOT_PENALTY_CAP =        -1 # Maximum penalty per step for this

        if self.steps_since_last_dot > NO_DOT_PENALTY_START_STEP:
            no_dot_penalty = NO_DOT_PENALTY_FACTOR * (self.steps_since_last_dot - NO_DOT_PENALTY_START_STEP)
            reward += max(NO_DOT_PENALTY_CAP, no_dot_penalty)

        # 10. Penalty for Immediate Reversal (Anti-Oscillation)
        REVERSAL_PENALTY = -1.5
        action_map = {"l": 0, "r": 1, "u": 2, "d": 3}
        current_action_int = action_map.get(self.game_state.direction)

        if self.last_action is not None and current_action_int is not None:
            # Check if actions are opposite (l<->r or u<->d)
            is_opposite = abs(current_action_int - self.last_action) == 1 and current_action_int // 2 == self.last_action // 2
            # Apply penalty only if no points were gained (didn't just eat something)
            if is_opposite and points_gain <= 0:
                # Optional: make penalty harsher if also stuck or not moving towards dot?
                reward += REVERSAL_PENALTY

        return reward
    
    
    def close(self):
        pygame.quit()
        
if __name__ == "__main__":
    import pygame, time

    env = PacmanEnvironment(render=True)
    env.reset()
    env.current_gen = 9999 # Set to max gen for testing
    done = False

    key_mappings = {
        pygame.K_LEFT: 0,
        pygame.K_RIGHT: 1,
        pygame.K_UP: 2,
        pygame.K_DOWN: 3
    }

    last_action = None
    total_reward = 0.0
    
    while not done:
        action = last_action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                mapped = key_mappings.get(event.key, None)
                if mapped is not None:
                    action = mapped

        if action is not None:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            last_action = action
            print(reward)

        if env.render_enabled:
            env.screen.fill((0, 0, 0))
            env.screen_manager.draw_screens()
            env.all_sprites.draw(env.screen)
            pygame.display.flip()
            
        time.sleep(1/60)
    print("Game Over, total reward:", total_reward)

    env.close()
    pygame.quit()
