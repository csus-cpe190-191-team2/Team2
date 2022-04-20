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
        lane_curve = lane.get_curve()  # 0==left_radii,1==right_radii,2==left_angle,3==right_angle,4==center
        print(lane_curve)

        lane.display(lane_curve[2], lane_curve[3], lane_curve[4])
    motor.destroy()
