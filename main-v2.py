import lane_finder as lf
# import lane_detection as lf
import device_finder as df
import motor2 as m
from time import sleep

def parse_command(curr_input, motor, lanes):
    if curr_input == "UP":
        motor.forward() #go forward
    if curr_input == "DOWN":
        motor.backward() #go backward
    if curr_input == "LEFT":
        motor.turn_left() #turn left
    if curr_input == "RIGHT":
        motor.turn_right() #turn right
    if curr_input == "A":
        lane_curve = lanes.get_curve()

        lanes.display(lane_curve[2], lane_curve[3])
        #motor.default_duty() #default speed
    if curr_input == "X":
        motor.toggle_state() #turn on/off
    if curr_input == "Y":
        motor.speed_up() #increase speed
    if curr_input == "B":
        lanes.get_img(display=True)
        lanes.draw_triangle(50,100)
        #motor.speed_down() #decrease speed
    if curr_input == "SELECT":
        motor.toggle_auto() #switch modes
    if curr_input == "START":
        motor.loop = False #stop program
        m.destroy()
    if curr_input == "Ltrigger":
        motor.rotate_left() #rotate left
    if curr_input == "Rtrigger":
        motor.rotate_right() #rotate right

if __name__ == '__main__':
    print("Configuring Controller...")
    inputDev = df.Controller(1)
    #m.destroy()
    motor = m.MotorControl()
    lanes = lf.LaneFinder()
    #inputDev.loop_listen()
    while motor.loop:
        #print("looping...")
        command = inputDev.read_command()
        if not command == None:
            print(command)
        parse_command(command, motor, lanes)
        #sleep(0.5)
        if motor.auto:
            pass
        else:
            pass
    inputDev.kill_device()
    quit()