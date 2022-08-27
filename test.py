import motor
import lane_detect as ld
import time

if __name__ == '__main__':  # Program start from here
    # code
    turn_deg = 15
    outer_lane_deg = 20
    inner_lane_deg = 30
    try:
        motor_ctl = motor.MotorControl()
        lane = ld.LaneDetect()

        # Camera warm up
        print("Warming up camera...")
        for x in range(5, 0, -1):
            print("Starting in: ", x)
            lane.get_img()
            time.sleep(1)

        # Write values to csv
        labels = "left_rad,right_rad,left_angle,right_angle,center,error\n"
        with open("lane.csv", "w") as f:
            f.write(labels)

        while True:
            lane_curve = lane.get_curve()
            # Appending to file
            with open("lane.csv", "a") as f:
                left_rad = "{:.3f}".format(lane_curve[0])
                right_rad = "{:.3f}".format(lane_curve[1])
                left_angle = "{:.3f}".format(lane_curve[2])
                right_angle = "{:.3f}".format(lane_curve[3])
                center = "{:.1f}".format(lane_curve[4])
                error = "{:.0f}".format(lane_curve[5])
                lane_vars = left_rad + ',' + right_rad + ',' + left_angle + ',' + \
                            right_angle + ',' + center + ',' + error + '\n'
                f.write(lane_vars)

            print(lane_curve)

            if lane_curve[5]:
                if lane_curve[2] > lane_curve[3]:
                    motor_ctl.set_speed(1)
                    motor_ctl.turn_right()
                    print("Turn Right")
                elif lane_curve[3] > lane_curve[2]:
                    motor_ctl.set_speed(1)
                    motor_ctl.turn_left()
                    print("Turn Left")
                else:
                    motor_ctl.set_speed(1)
                    motor_ctl.turn_right()
                    print("Turn Right")
            else:
                if lane_curve[4] != 0:
                    if lane_curve[4] > 1:
                        if lane_curve[3] > inner_lane_deg:
                            motor_ctl.set_speed(1)
                            motor_ctl.turn_right()
                            print("Turn Right")
                            # else:
                            #     motor_ctl.set_speed(1)
                            #     motor_ctl.turn_left()
                            #     print("Turn Left")
                        else:
                            motor_ctl.set_speed(0)
                            motor_ctl.turn_left()
                            motor_ctl.stop()
                            print("Rotate Left")
                    elif lane_curve[4] < -1:
                        if lane_curve[2] > outer_lane_deg:
                            motor_ctl.set_speed(1)
                            motor_ctl.turn_left()
                            print("Turn Left")
                            # else:
                            #     motor_ctl.set_speed(1)
                            #     motor_ctl.turn_right()
                            #     print("Turn Right")
                        else:
                            motor_ctl.set_speed(0)
                            motor_ctl.turn_right()
                            motor_ctl.stop()
                            print("Rotate Right")
                    else:
                        print("Move forward")
                        motor_ctl.set_speed(1)
                        motor_ctl.forward()

                else:
                    print("Move forward")
                    motor_ctl.set_speed(1)
                    motor_ctl.forward()

            lane.display(lane_curve[2], lane_curve[3], lane_curve[4], display=True)
    finally:
        motor.destroy()
