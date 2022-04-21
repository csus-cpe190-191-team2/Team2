import motor
import lane_detect as ld
import time

if __name__ == '__main__':  # Program start from here
    # code
    motor_ctl = motor.MotorControl()
    lane = ld.LaneDetect()
    lane.get_img()
    time.sleep(5)

    while True:
        lane_curve = lane.get_curve()
        print(lane_curve)
        if lane_curve[2] > 20 or lane_curve[3] > 20:
            if lane_curve[3] > 20:
                motor_ctl.set_speed(1)
                motor_ctl.turn_right()
            else:
                motor_ctl.set_speed(1)
                motor_ctl.turn_left()
        elif lane_curve[4] >= 1 or lane_curve[4] <= -1:
            if lane_curve[4] >= 1:
                motor_ctl.set_speed(0)
                motor_ctl.turn_left()
                print("Spin Left")
            else:
                motor_ctl.set_speed(0)
                motor_ctl.turn_right()
                print("Spin Right")
        else:
            print("Move forward")
            motor_ctl.set_speed(1)
            motor_ctl.forward()

        lane.display(lane_curve[2], lane_curve[3], lane_curve[4])

    motor.destroy()

