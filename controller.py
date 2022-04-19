import evdev
import motor


class Input:
    def __init__(self):
        self.gamepad = None
        self.deviceName = None
        self.button_map = {}
        self.motor_control = motor.MotorControl()
        path = "/dev/input"

        # Get list of currently connected input devices
        device_list = [evdev.InputDevice(path) for path in evdev.list_devices()]
        # Attach to device with keyboard input
        for device in device_list:
            if "Keyboard" in device.name or "keyboard" in device.name\
                    or "Controller" in device.name or "controller" in device.name:
                self.gamepad = evdev.InputDevice(device)
                self.deviceName = device
        if self.gamepad is not None:
            print("Connected to: ", self.gamepad)
        else:
            print("No Bluetooth Device Connected!")
            motor.destroy()

    # Handle controller input
    def input_listen(self):
        curr_input = None  # Temp var holds current input value

        ipega_mapping = {
            104: "UP",
            109: "DOWN",
            106: "LEFT",
            107: "RIGHT",
            37: "A",
            51: "X",
            24: "Y",
            38: "B",
            20: "SELECT",
            22: "START",
            17: "LEFT TRIGGER",
            26: "RIGHT TRIGGER"
        }

        eightbitdo_mapping = {
            47: "UP",
            33: "DOWN",
            19: "LEFT",
            34: "RIGHT",
            35: "A",
            36: "X",
            24: "Y",
            37: "B",
            50: "SELECT",
            25: "START",
            38: "LEFT TRIGGER",
            51: "RIGHT TRIGGER"
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

        # Assign button map to appropriate input device
        if "Bluetooth Gamepad Keyboard" in self.deviceName.name:
            self.button_map = ipega_mapping
        elif "8BitDo" in self.deviceName.name:
            self.button_map = eightbitdo_mapping
        elif "Controller" in self.deviceName.name:
            self.button_map = ps5_mapping
        else:
            print("Device does not match any recognized button map...")
            quit()