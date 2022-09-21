import lane_detect_CNN as cnn
import motor as m
import eyes as E
import ultrasonic_distance as dist
import controller as c


def parse_command(motor, command):
    if motor.auto:  # cnn is controlling:
        if command == "X":
            motor.toggle_motor()  # turn on/off
        if command == "SELECT":
            motor.toggle_auto()  # switch modes
        if command == "START":
            motor.loop = False  # stop program
    else:   # Manual control
        if command == "UP":
            motor.forward()  # go forward
        if command == "DOWN":
            motor.backward()  # go backward
        if command == "LEFT":
            motor.turn_left()  # turn left
        if command == "RIGHT":
            motor.turn_right()  # turn right
        if command == "A":
            motor.set_speed(1) #default speed
        if command == "X":
            motor.toggle_motor()  # turn on/off
        if command == "Y":
            motor.speed_up()  # increase speed
        if command == "B":
            motor.speed_down() #decrease speed
        if command == "SELECT":
            motor.toggle_auto()  # Switch modes
            motor.set_speed(1)   # Default speed
        if command == "START":
            motor.loop = False  # stop program
        if command == "LEFT TRIGGER":
            motor.rotate_left()  # rotate left
        if command == "RIGHT TRIGGER":
            motor.rotate_right()  # rotate right


if __name__ == '__main__':
    stop_dist = 8   # Stopping distance in cm
    try:
        with open('vars/eva_ascii') as f:
            print(f.read())
        print("Getting ready for takeoff...")
        auto_motor = cnn.DriveDetection()
        eye = E.Eyes()
        driver = m.MotorControl()
        controller = c.Controller()
        print("Initialized!")
        while driver.loop:
            #check input no matter what
            command = controller.read_command()
            parse_command(driver, command)
            if command is not None:
                if command == 'SELECT':
                    print(command, '--->', f'Self Driving: {driver.auto}')
                elif command == 'START':
                    print("Exiting...")
                elif (command == 'Y' or command == 'B' or command == 'A') and driver.drive_state:
                    if command == 'Y':
                        print(command, '--->', f'Speed UP: {driver.current_duty}')
                    elif command == 'B':
                        print(command, '--->', f'Speed DOWN: {driver.current_duty}')
                    else:
                        print(command, '--->', f'Speed DEFAULT: {driver.current_duty}')
                else:
                    print(command, '--->', driver.get_drive_state_label())

            # Check for obstruction
            if dist.distance() < stop_dist:
                print("OBSTRUCTION DETECTED")   # DEBUG output
                if driver.drive_state:
                    driver.toggle_motor()
                while dist.distance() < stop_dist:
                    continue
                print("Continuing...")  # DEBUG output
                driver.toggle_motor()
            else:
                # If in auto then let cnn predict
                if driver.auto:
                    img = eye.get_thresh_img()
                    motor_pred, cert = auto_motor.drive_predict(img)
                    print(motor_pred[1])
                    if motor_pred[1] == 'forward':
                        driver.forward()
                    if motor_pred[1] == 'left':
                        driver.turn_left()
                    if motor_pred[1] == 'right':
                        driver.turn_right()
                    if motor_pred[1] == 'rotate_left':
                        driver.rotate_left()
                    if motor_pred[1] == 'rotate_right':
                        driver.rotate_right()
    except Exception as e:
        print(e)
    finally:
        try:
            eye.destroy()
            controller.kill_device()
        except Exception as e:
            print(e)
        m.destroy()


