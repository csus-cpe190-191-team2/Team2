# import controller
import motor
import os
import lane_detect as ld
import time

if __name__ == '__main__':  # Program start from here
    # code
    motor_ctl = motor.MotorControl()
    lane = ld.LaneDetect()
    lane.get_img()
    time.sleep(3)
    auto = False
    print("Configuring Keyboard Controller Input...")
    # Select current input device based on what is connected
    print("bbb")
    # inputDev = controller.Input()
    print("test")
    while True:
        lane_curve = lane.get_curve()
        print(lane_curve)
        if lane_curve[4] > 0:
            if lane_curve[2] > lane_curve[3]:
                if lane_curve[2] > 30:
                    motor_ctl.turn_left()
                    print("Turn Left")
                else:
                    motor_ctl.set_speed(0)
                    motor_ctl.turn_left()
                    print("Spin Left")
        elif lane_curve[4] < 0:
            if lane_curve[3] > lane_curve[2]:
                if lane_curve[3] > 30:
                    motor_ctl.turn_right()
                    print("Turn Right")
            else:
                motor_ctl.set_speed(0)
                motor_ctl.turn_right()
                print("Spin Right")
        else:
            print("Move forward")
            motor_ctl.forward()
            motor_ctl.set_speed(1)
            print(motor_ctl.left_duty, ' ', motor_ctl.left_duty)
        lane.display(lane_curve[2], lane_curve[3], lane_curve[4])
    motor.destroy()

