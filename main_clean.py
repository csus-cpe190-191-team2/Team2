import controller
import motor_handle
import os
import processing

if __name__ == '__main__':  # Program start from here
    # code
    try:
        print("Configuring Keyboard Controller Input...")
        # Select current input device based on what is connected
        inputDev = controller.Input()
        manual = motor_handle.M_motor_react()
        auto = motor_handle.A_motor_react()
        processor = processing.Brain()
        drive_state = True #true is manual, false is auto
        # If input device is connected begin listening for input
        if inputDev.gamepad is not None:
            inputDev.input_listen()#change name of function
            try:
                for event in inputDev.gamepad.read_loop():
                    if event.value == 1:  # 1 means button is pressed, 0 is released
                        curr_input = inputDev.button_map.get(event.code)
                        # Exit manual mode if START button is pressed
                        if curr_input == "START":
                            if(drive_state): drive_state = False
                            else: drive_state = True

                        if(drive_state): manual.handle_command(curr_input)
                        else:
                            if(!processing.handle_distance()):
                                curve[] = lane.get_curve()
                                object = getObjects()





            except OSError as e:
                print("Controller Disconnected: ", e)
                # Stop Motors
            except AttributeError as e:
                print("Button map configuration error: ", e)
            finally:
                motor.destroy()
                inputDev.gamepad.close()

    except KeyboardInterrupt:   # potentially unneeded exception handle
        print("Keyboard")
        # destroy()  # Will be useful cleanup function for later exception handling

    finally:
        print("clean up")
        # GPIO.cleanup()  # clean all gpio
        os._exit(0)