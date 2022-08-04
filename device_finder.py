import evdev
import asyncio
from evdev import InputDevice, categorize, ecodes

path = "/dev/input"

async def print_events_loop(device):
    async for event in device.async_read_loop():
        print(device.path, evdev.categorize(event), sep=': ')

async def print_events_single(device):
    event = device.async_read_one()
    print(device.path, evdev.categorize(event), sep=': ')

class Controller:
    def __init__(self, dev=0):
        self.gamepad = None
        self.device_name = None
        self.map = {}
        self.set_controller(dev)
        self.set_map()

    def kill_device(self):
        self.gamepad.close()

    def set_controller(self, dev=0):
        device_list = [evdev.InputDevice(path) for path in evdev.list_devices()]
        if dev == 0:
            i = 1
            for device in device_list:
                print(i, device)
                i += 1
            x = input('Device #: ')
            i = 1
            for device in device_list:
                if int(x) == i:
                    self.gamepad = evdev.InputDevice(device)
                    self.device_name = device.name
                i += 1
        elif dev == 1:
            for device in device_list:
                if "8BitDo" in device.name:
                    self.gamepad = evdev.InputDevice(device)
                    self.device_name = device.name
            if self.gamepad == None:
                print("8BitDo not found...exiting")
                quit()
        elif dev == 2:
            for device in device_list:
                if "Keyboard" in device.name:
                    self.gamepad = evdev.InputDevice(device)
                    self.device_name = device.name
            if self.gamepad == None:
                print("Keyboard not found...exiting")
                quit()
        print("Connected to: ", self.gamepad)

    def get_codes(self):
        for event in self.gamepad.read_loop():
            if event.type == ecodes.EV_KEY:
                print(event)

    def set_map(self):
        print("Acquiring map...")
        eightbitdo_mapping = {
            46: "UP",
            32: "DOWN",
            18: "LEFT",
            33: "RIGHT",
            34: "A",
            35: "X",
            23: "Y",
            36: "B",
            49: "SELECT",
            24: "START",
            37: "LEFT TRIGGER",
            50: "RIGHT TRIGGER"
        }
        keyboard_mapping = {
            17: "UP", #W
            31: "DOWN", #S
            30: "LEFT", #A
            32: "RIGHT", #D
            36: "A", #J
            23: "X", #I
            22: "Y", #U
            37: "B", #K
            50: "SELECT", #M
            16: "START", #Q
            38: "LEFT TRIGGER", #L
            19: "RIGHT TRIGGER" #R
        }
        ps5_mapping = {
            16: "UP",
            18: "DOWN",
            15: "LEFT",
            17: "RIGHT",
            306: "A",
            305: "X",
            308: "Y",
            307: "B",
            313: "SELECT",
            314: "START",
            309: "LEFT TRIGGER",
            310: "RIGHT TRIGGER"
        }
        if "Keyboard &" in self.device_name:
            self.map = keyboard_mapping
            print("Keyboard mapped")
        elif "8BitDo" in self.device_name:
            self.map = eightbitdo_mapping
            print("8BitDo mapped")
        elif "Controller" in self.device_name:
            self.map = ps5_mapping
        else:
            print("Device does not match any recognized button map...")
            quit()

    def loop_listen(self):
        asyncio.ensure_future(print_events_loop(self.gamepad))
        loop = asyncio.get_event_loop()
        loop.run_forever()
        print("listening...")

    def single_listen(self):
        asyncio.ensure_future(print_events_loop(self.gamepad))

    def read_command(self):
        event = self.gamepad.read_one()
        if event == None:
            return None
        if event.value == 1:
            curr_input = self.map.get(event.code)
            return curr_input

# controller = Controller(2)
# controller.get_codes()
# controller.kill_device()
# print("Done")
# quit()