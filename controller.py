import evdev
# from evdev import InputDevice,categorize,ecodes


class Input:
    def __init__(self):
        self.gamepad = None
        self.deviceName = None
        path = "/dev/input"

        # Get list of currently connected input devices
        device_list = [evdev.InputDevice(path) for path in evdev.list_devices()]
        
        # Attach to device with keyboard input
        for device in device_list:
            print(device.name)
            name = device.name
            string1 = 'Keyboard'
            string2 = 'keyboard'
            if string1 in name or string2 in name:
                self.gamepad = evdev.InputDevice(device)
                self.deviceName = device
        if self.gamepad is not None:
            print("Connected to: ", self.gamepad)
        else:
            print("No Bluetooth Device Connected!")

    # Handle controller input
    def input_listen(self):
        curr_input = None        # Temp var holds current input value
        button_map = None    # Adjust button map based on currently connected input device

        ipega_mapping = {
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

        eightbitdo_mapping = {
            46 : "UP",
            32 : "DOWN",
            18 : "LEFT",
            33 : "RIGHT",
            34 :  "A",
            35 :  "X",
            23 :  "Y",
            36 :  "B",
            49 :  "SELECT",
            24 :  "START",
            37 :  "LEFT TRIGGER",
            50 :  "RIGHT TRIGGER"
        }

        # Assign button map to appropriate input device
        if "Bluetooth Gamepad Keyboard" in self.deviceName.name:
            button_map = ipega_mapping
        elif "8BitDo" in self.deviceName.name:
            button_map = eightbitdo_mapping
        else:
            print("Device does not match any recognized button map...")
            quit()

        # Begin reading in input
        try:
            for event in self.gamepad.read_loop():
                # print("Event Type: ", event.type, "\tEvent Code: ", event.code, '\tEvent Value: ', event.value)# DEBUG
                if event.value == 1:    # 1 means button is pressed, 0 is released
                    curr_input = button_map.get(event.code)
                    if curr_input == "START":
                        self.gamepad.close()
                        quit()
                    print(curr_input)    # Temp output for DEBUG, eventually handle motor control here...
        except OSError as e:
            print("Controller Disconnected: ", e)
            # Stop Motors
        except AttributeError as e:
            print("Button map configuration error: ", e)
