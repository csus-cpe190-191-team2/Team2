import evdev
#from evdev import InputDevice,categorize,ecodes

class Input:
    def __init__(self):
        path = "/dev/input"
        self.gamepad = None
        self.deviceName = None

        # Get list of currently connected input devices
        device_list = [evdev.InputDevice(path) for path in evdev.list_devices()]

        # Attach to device with keyboard input
        for device in device_list:
            if "Keyboard" or "keyboard" in device.name:
                self.gamepad = evdev.InputDevice(device)
                self.deviceName = device
        if self.gamepad is not None:
            print("Connected to: ", self.gamepad)
        else:
            print("No Bluetooth Device Connected!")

    # Handle controller input
    def input_listen(self):
        currInput = None        # Temp var holds current input value
        currButtonMap = None    # Adjust button map based on currently connected input device

        ipegaButtonMap = {
            103: "UP",
            108: "DOWN",
            105: "LEFT",
            106: "RIGHT",
            36:  "A",
            50:  "X",
            23:  "Y",
            37:  "B",
            19:  "SELECT",
            21:  "START",
            16:  "LEFT TRIGGER",
            25:  "RIGHT TRIGGER"
        }

        eightBitDoButtonMap = {
            "up": "UP",
            "down": "DOWN",
            "left": "LEFT",
            "right": "RIGHT",
            "a":  "A",
            "x":  "X",
            "y":  "Y",
            "b":  "B",
            "select":  "SELECT",
            "start":  "START",
            "Ltrigger":  "LEFT TRIGGER",
            "Rtrigger":  "RIGHT TRIGGER"
        }

        # Assign button map to appropriate input device
        if "Bluetooth Gamepad Keyboard" in self.deviceName.name:
            currButtonMap = ipegaButtonMap
        elif "8bitdo" in self.deviceName.name:
            currButtonMap = eightBitDoButtonMap
        else:
            print("Device does not match any recognized button map...")

        # Begin reading in input
        try:
            for event in self.gamepad.read_loop():
                #print("Event Type: ", event.type, "\tEvent Code: ", event.code, '\tEvent Value: ', event.value)    # DEBUG
                if event.value == 1:    # 1 means button is pressed, 0 is released
                    currInput = currButtonMap.get(event.code)
                    if currInput == "START":
                        self.gamepad.close()
                        quit()
                    print(currInput)    # Temp output for DEBUG, eventually handle motor control here...
        except OSError as e:
            print("Controller Disconnected: ", e)
        except AttributeError as e:
            print("Button map configuration error: ", e)

