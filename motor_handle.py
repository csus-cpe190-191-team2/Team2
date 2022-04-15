import motor

class M_motor_react:
    def __init__(self):
        self.motor_control = motor.MotorControl()

    def handle_command(self, curr_input):#does self and curr_input need to be switched?
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
            print("Changed back to manual...")
        if curr_input == "Ltrigger":
            print("No Function Yet...")
            #  Do something?
        if curr_input == "Rtrigger":
            print("No Function Yet...")
            #  Do something?

class A_motor_react:
    def __init__(self):
        self.motor_control = motor.AutoControl()

    def move(self, value):
        self.motor_control.move(value)
