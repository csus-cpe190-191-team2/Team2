import evdev
import motor
# from evdev import InputDevice,categorize,ecodes


class Input:
    def __init__(self):
        self.gamepad = None
        self.deviceName = None
        self.motor_control = motor.MotorControl()
        path = "/dev/input"

        # Get list of currently connected input devices
        device_list = [evdev.InputDevice(path) for path in evdev.list_devices()]

        # Attach to device with keyboard input
        for device in device_list:
            if "Keyboard" in device.name or "keyboard" in device.name:
                self.gamepad = evdev.InputDevice(device)
                self.deviceName = device
        if self.gamepad is not None:
            print("Connected to: ", self.gamepad)
        else:
            print("No Bluetooth Device Connected!")
            motor.destroy()

    # Handle controller input
    def input_listen(self):
        curr_input = None        # Temp var holds current input value
        button_map = None        # Adjust button map based on currently connected input device

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
            46: "UP",
            32: "DOWN",
            18: "LEFT",
            33: "RIGHT",
            34:  "A",
            35:  "X",
            23:  "Y",
            36:  "B",
            49:  "SELECT",
            24:  "START",
            37:  "LEFT TRIGGER",
            50:  "RIGHT TRIGGER"
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

                    print(curr_input)    # Temp output for DEBUG
                    if curr_input == "UP":
                        self.motor_control.forward()
                    if curr_input == "DOWN":
                        self.motor_control.backward()
                    if curr_input == "LEFT":
                        self.motor_control.turn_left()
                    if curr_input == "RIGHT":
                        self.motor_control.turn_right()
                    if curr_input == "A":
                        self.motor_control.speed_up()
                    if curr_input == "X":
                        self.motor_control.speed_down()
                    if curr_input == "Y":
                        self.motor_control.stop()
                    if curr_input == "B":
                        print("No Function Yet...")
                        #  Do something?
                    if curr_input == "SELECT":
                        print("No Function Yet...")
                        #  Do something?
                    # Exit manual mode if START button is pressed
                    if curr_input == "START":
                        self.motor_control.stop()
                        motor.destroy()
                        self.gamepad.close()
                        quit()
                    if curr_input == "Ltrigger":
                        print("No Function Yet...")
                        #  Do something?
                    if curr_input == "Rtrigger":
                        print("No Function Yet...")
                        #  Do something?

                    # Temp DEBUG output
                    print("Left: ", self.motor_control.left_duty, "\tRight: ", self.motor_control.right_duty)

        except OSError as e:
            print("Controller Disconnected: ", e)
            # Stop Motors
        except AttributeError as e:
            print("Button map configuration error: ", e)
        finally:
            motor.destroy()
            self.gamepad.close()
