from pyboy import PyBoy, WindowEvent

class Controller:
    def __init__(self, rom_path):
        self.pyboy = PyBoy(rom_path, window_scale=1)
        self.pyboy.set_emulation_speed(target_speed=0)
        self.movements = ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]

        self.event_dict_press = {
                "UP": WindowEvent.PRESS_ARROW_UP,
                "DOWN": WindowEvent.PRESS_ARROW_DOWN,
                "LEFT": WindowEvent.PRESS_ARROW_LEFT,
                "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
                "A": WindowEvent.PRESS_BUTTON_A,
                "B": WindowEvent.PRESS_BUTTON_B,
                "START": WindowEvent.PRESS_BUTTON_START,
                "SELECT": WindowEvent.PRESS_BUTTON_SELECT
            }

        self.event_dict_release = {
                "UP": WindowEvent.RELEASE_ARROW_UP,
                "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
                "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
                "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
                "A": WindowEvent.RELEASE_BUTTON_A,
                "B": WindowEvent.RELEASE_BUTTON_B,
                "START": WindowEvent.RELEASE_BUTTON_START,
                "SELECT": WindowEvent.RELEASE_BUTTON_SELECT
            }
        

    def handleMovement(self, movement, ticks_per_input=30, wait=60):
        self.pyboy.send_input(self.event_dict_press[movement])
        [self.pyboy.tick() for _ in range(ticks_per_input)]
        self.pyboy.send_input(self.event_dict_release[movement])
        [self.pyboy.tick() for _ in range(wait)]

    def screen_image(self):
        return self.pyboy.botsupport_manager().screen().screen_image()
    
    def get_memory_value(self, address):
        return self.pyboy.get_memory_value(address)
    
    def screen_size(self):
        return self.pyboy.botsupport_manager().screen().screen_ndarray().shape[:2]
    
    def stop(self, save=True):
        self.pyboy.stop(save)