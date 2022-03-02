import controller
import os

if __name__ == '__main__':  # Program start from here
    # code
    try:
        print("Configuring Keyboard Controller Input...")
        # Select current input device based on what is connected
        inputDev = controller.Input()
        # If input device is connected begin listening for input
        if inputDev.gamepad is not None:
            inputDev.input_listen()

    except KeyboardInterrupt:   # potentially unneeded exception handle
        print("Keyboard")
        # destroy()  # Will be useful cleanup function for later exception handling

    finally:
        print("clean up")
        # GPIO.cleanup()  # clean all gpio
        os._exit(0)
