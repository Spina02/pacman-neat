import os
import pygame
import time
import sys
# from pacman.src.game.state_management import GameState
# from pacman.src.game.event_management import EventHandler
# # Import the new "headless" screen manager or the existing one with a toggle
# from pacman.src.gui.screen_management import ScreenManager

pacman_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "pacman"))
print("Pacman path:", pacman_path)
if pacman_path not in sys.path:
    sys.path.insert(0, pacman_path)
    from src.game.state_management import GameState
    from src.game.event_management import EventHandler
    from src.gui.screen_management import ScreenManager


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

    def step(self, action):
        """
        Advance one frame with the specified action.
        Args:
            action (int): Action chosen by NEAT: e.g. 0=left, 1=right, 2=up, 3=down, ...
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

        # We need to replicate your "GameRun.main()" loop but only for 1 iteration:
        # We'll call the event handler manually for any QUIT events, etc.
        # Then call the drawing/updating code once.

        for event in pygame.event.get():
            self.event_handler.handle_events(event)
        # if not self.game_state.running:
        #     # If the user closed the game, treat it as a done (if that is desired).
        #     done = True

        # If rendering, fill the screen, draw, etc.
        if self.render_enabled:
            self.screen.fill((0,0,0))
            self.screen_manager.draw_screens()
            self.all_sprites.draw(self.screen)

        # Actually update game logic
        dt = 1  # You can fix dt=1 or mimic frames (like in your runner code)
        self.all_sprites.update(dt)

        # Gather the immediate reward
        # For example, you can use the difference in scoreboard from last step,
        # or base it on surviving, etc.
        reward = self.calculate_reward()

        # Determine if done
        # E.g. if Pacman is dead or the level is complete
        if self.game_state.is_pacman_dead or self.game_state.level_complete:
            done = True

        # Construct next observation
        obs = self.get_observation()

        # Info dict can hold debugging info if desired
        info = {}

        # If rendering, flip the display
        if self.render_enabled:
            pygame.display.flip()

        return obs, reward, done, info

    def get_observation(self):
        """
        Return the current game state in a numeric form 
        that NEAT can work with (list, tuple, or np.array).
        For instance, Pacman position, ghost positions, 
        current score, distance to nearest dot, etc.
        """
        # Example: get pacman coords and ghost coords
        # Pacman coords might be self.game_state.pacman_rect 
        # or self.game_state.pacman_pos
        # You can feed them into a normalized vector
        return [0.0] * self.observation_size  # placeholder

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
    config = { "game": { "observation_size": 10, # dimensione dell'osservazione (placeholder) 
                        "action_size": 4, # 4 azioni: left, right, up, down
                        "render": False # headless per test veloci 
                        } }
    # (Opzionale) se vuoi testare in modalità headless, puoi impostare SDL_VIDEODRIVER
    if not config["game"]["render"]:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    # Crea l'environment e resettalo
    env = PacmanEnvironment(config)
    obs = env.reset()
    print("Initial observation:", obs)

    # Esegui alcuni step con azioni predefinite (qui ad esempio tutte l'azione 0, cioè "left")
    num_steps = 50
    for step in range(num_steps):
        # In questo test usiamo sempre la stessa azione, 
        # ma potresti anche randomizzare l'azione in base a config["game"]["action_size"]
        obs, reward, done, info = env.step(0)
        print(f"Step {step}: reward = {reward}, done = {done}")
        if done:
            print("Episode finished at step", step)
            break
        # (Opzionale) breve pausa per simulare il tempo tra gli step
        time.sleep(0.1)

    env.close()
    pygame.quit()
    print("Test completato.")