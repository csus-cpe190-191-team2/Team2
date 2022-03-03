import controller
import motor
import os

if __name__ == '__main__':  # Program start from here
    # code
    try:
        print("Configuring Keyboard Controller Input...")
        # Select current input device based on what is connected
        print("bbb")
        inputDev = controller.Input()
        print("test")
        # If input device is connected begin listening for input
        if inputDev.gamepad is not None:
            inputDev.input_listen()#change name of function
            try:
                for event in inputDev.gamepad.read_loop():
                    # print("Event Type: ", event.type, "\tEvent Code: ", event.code, '\tEvent Value: ', event.value)# DEBUG
                    if event.value == 1:    # 1 means button is pressed, 0 is released
                        curr_input = inputDev.button_map.get(event.code)

                        print(curr_input)    # Temp output for DEBUG
                        if curr_input == "UP":
                            inputDev.motor_control.forward()
                        if curr_input == "DOWN":
                            inputDev.motor_control.backward()
                        if curr_input == "LEFT":
                            inputDev.motor_control.turn_left()
                        if curr_input == "RIGHT":
                            inputDev.motor_control.turn_right()
                        if curr_input == "A":
                            inputDev.motor_control.speed_up()
                        if curr_input == "X":
                            inputDev.motor_control.speed_down()
                        if curr_input == "Y":
                            inputDev.motor_control.stop()
                        if curr_input == "B":
                            print("No Function Yet...")
                            #  Do something?
                        if curr_input == "SELECT":
                            print("No Function Yet...")
                            #  Do something?
                        # Exit manual mode if START button is pressed
                        if curr_input == "START":
                            inputDev.motor_control.stop()
                            inputDev.gamepad.close()
                            quit()
                        if curr_input == "Ltrigger":
                            print("No Function Yet...")
                            #  Do something?
                        if curr_input == "Rtrigger":
                            print("No Function Yet...")
                            #  Do something?

                        # Temp DEBUG output
                        print("Left: ", inputDev.motor_control.left_duty, "\tRight: ", inputDev.motor_control.right_duty)

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
