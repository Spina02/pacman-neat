import os
import sys
import pygame

current_dir = os.path.dirname(__file__)
super_mario_path = os.path.join(current_dir, "super_mario")
if super_mario_path not in sys.path:
    sys.path.append(super_mario_path)

from classes.Dashboard import Dashboard
from classes.Level import Level
from classes.Menu import Menu
from classes.Sound import Sound
from entities.Mario import Mario

class GameEngine:
    def __init__(self, headless=False):
        """
        Inizializza il motore di gioco.
        Se headless è True, si evita l'apertura di una finestra grafica.
        """
        # Se in modalità headless, setta il driver video di SDL su "dummy"
        self.headless = headless
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        # Preinizializzazione audio e inizializzazione pygame
        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()

        self.windowSize = (640, 480)
        if not self.headless:
            self.screen = pygame.display.set_mode(self.windowSize)
        else:
            # In modalità headless, usiamo una surface dummy
            self.screen = pygame.Surface(self.windowSize)

        self.clock = pygame.time.Clock()
        self.max_frame_rate = 60

        # Inizializza gli oggetti del gioco
        self.dashboard = Dashboard("./img/font.png", 8, self.screen)
        self.sound = Sound()
        self.level = Level(self.screen, self.sound, self.dashboard)
        self.menu = Menu(self.screen, self.dashboard, self.level, self.sound)
        self.mario = None  # Sarà creato dopo il menu

    def reset(self):
        """
        Resetta lo stato del gioco.
        Attende la fine del menu (o la selezione) e crea una nuova istanza di Mario.
        """
        # Loop del menu fino a quando l'utente non inizia il gioco
        while not self.menu.start:
            self.menu.update()
            if not self.headless:
                pygame.display.update()
            self.clock.tick(self.max_frame_rate)
        # Crea un nuovo oggetto Mario
        self.mario = Mario(0, 0, self.level, self.screen, self.dashboard, self.sound)

    def step(self, action):
        """
        Esegue un tick di aggiornamento del gioco in base all'azione fornita.
        :param action: dizionario (o altra struttura) che indica il comando, ad esempio:
                       {'move': 'left', 'jump': True}
        """
        # Se Mario ha un metodo per gestire gli input (es. handle_action), lo richiamiamo:
        if self.mario is not None and hasattr(self.mario, 'handle_action'):
            self.mario.handle_action(action)
        # Altrimenti, potresti aggiornare direttamente gli attributi di Mario,
        # ad esempio: self.mario.move_left() oppure self.mario.jump()

        # Aggiorna lo stato del gioco
        if self.mario.pause:
            self.mario.pauseObj.update()
        else:
            self.level.drawLevel(self.mario.camera)
            self.dashboard.update()
            self.mario.update()

        # Se non siamo in modalità headless, aggiorna il display
        if not self.headless:
            pygame.display.update()
        self.clock.tick(self.max_frame_rate)

    def get_state(self):
        """
        Restituisce uno stato sintetico del gioco da utilizzare come input per NEAT.
        Esempio di features: posizione e (eventualmente) velocità di Mario, punteggio, etc.
        """
        state = {}
        if self.mario is not None:
            # Assumiamo che Mario abbia un rettangolo (rect) per la posizione
            state['mario_x'] = self.mario.rect.x
            state['mario_y'] = self.mario.rect.y
            # Se definito, aggiungi altre features (ad esempio la velocità)
            if hasattr(self.mario, 'velocity'):
                state['mario_velocity'] = self.mario.velocity
            # Aggiungi altre informazioni utili, come il punteggio corrente
            state['score'] = getattr(self.dashboard, 'score', 0)
            # Puoi estendere lo stato includendo, ad esempio, la distanza da eventuali nemici
        return state

    def render(self):
        """
        Visualizza il gioco se non siamo in modalità headless.
        In modalità play, questa funzione si occupa di aggiornare la finestra grafica.
        """
        if not self.headless:
            pygame.display.update()


# Esempio di test per il GameEngine (modalità play)
if __name__ == "__main__":
    engine = GameEngine(headless=False)
    engine.reset()  # Avvia il menu e poi crea Mario
    running = True

    while running:
        # Gestione degli eventi per poter chiudere la finestra
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Per test, eseguiamo un tick senza input (puoi sostituire con un dizionario contenente un'azione)
        engine.step(action={})
        engine.render()

    pygame.quit()
