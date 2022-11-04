import lane_detect_CNN as cnn
import motor as m
import eyes as E
import controller as c
import ultrasonic_distance as dist
from threading import Thread, current_thread, Event
from multiprocessing import current_process
import time
import os

stop_dist = 10    # cm
stop_event = Event()
drive_auto_event = Event()


def io_thread(controller: c, driver: m):
    timeout = 0.001  # ms timeout (range: ~1-300ms)
    pid = os.getpid()
    threadName = current_thread().name
    processName = current_process().name
    print(f"{pid}/{processName}/{threadName} ---> Starting IO Thread...")
    last_event = time.time()
    while not stop_event.is_set():
        if drive_auto_event.is_set():
            if (time.time()-last_event) > timeout:
                command = controller.read_command()
                if command is not None:
                    parse_command(driver, command, True)
                last_event = time.time()
            else:
                time.sleep(timeout)
        else:
            command = controller.read_command()
            if command is not None:
                parse_command(driver, command, True)

    print(f"{pid}/{processName}/{threadName} ---> Finished IO Thread...")


def parse_command(driver, command, print_activity=False):
    # print(command)                # DEBUG
    if command == "START":          # stop program
        driver.loop = False
        stop_event.set()
        print("Exiting...")
    elif command == "X":            # Toggle motor
        driver.toggle_motor()
        if print_activity:
            print(command, '--->', driver.get_drive_state_label(),
                  f'---> Motors: {"On" if driver.motor_state else "Off"}')
    elif driver.auto:               # CNN is controlling:
        if command == "SELECT":     # switch driving modes
            driver.motor_off()
            driver.toggle_auto()
            drive_auto_event.clear()
            if print_activity:
                print(command, '--->', f'Self Driving: {driver.auto}')
    else:                               # Manual control
        if command == "UP":             # go forward
            driver.forward()
        if command == "DOWN":           # go backward
            driver.backward()
        if command == "LEFT":           # turn left
            driver.turn_left()
        if command == "RIGHT":          # turn right
            driver.turn_right()
        if command == "A":              # default speed
            driver.set_speed(1)
        if command == "Y":              # increase speed
            driver.speed_up()
        if command == "B":              # decrease speed
            driver.speed_down()
        if command == "LEFT TRIGGER":   # rotate left
            driver.rotate_left()
        if command == "RIGHT TRIGGER":  # rotate right
            driver.rotate_right()
        if command == "SELECT":         # switch driving modes
            E.cv2.destroyAllWindows()
            drive_auto_event.set()
            driver.motor_on()
            driver.set_speed(1)         # Default speed
            driver.toggle_auto()

        if print_activity:
            if driver.motor_state:
                if command == 'Y':
                    print(command, '--->', f'Speed UP: {driver.current_duty}')
                elif command == 'B':
                    print(command, '--->', f'Speed DOWN: {driver.current_duty}')
                elif command == 'A':
                    print(command, '--->', f'Speed DEFAULT: {driver.current_duty}')
                else:
                    print(command, '--->', driver.get_drive_state_label())
            else:
                print(command, '--->', driver.get_drive_state_label(),
                      f'---> Motors: {"On" if driver.motor_state else "Off"}')


def auto_drive_handler(driver, command):
    if command == "forward":        # go forward
        driver.forward()
    if command == "left":           # turn left
        driver.turn_left()
    if command == "right":          # turn right
        driver.turn_right()
    if command == "rotate_left":    # rotate left
        driver.rotate_left()
    if command == "rotate_right":   # rotate right
        driver.rotate_right()


if __name__ == '__main__':
    try:    # Eeeee... va?
        with open('vars/eva_ascii') as f:
            print(f.read())
    except Exception as e:
        print(e)
    except FileExistsError as e:
        print("Eee... VA?")

    print("Preparing Launch Procedures...")
    controller = c.Controller(3)
    driver = m.MotorControl()
    driver.set_speed(1)
    eye = E.Eyes()
    auto_motor = cnn.DriveDetection()

    # Start gamepad i/o thread
    t1_io = Thread(target=io_thread, args=(controller, driver))
    t1_io.start()
    pid = os.getpid()
    threadName = current_thread().name
    processName = current_process().name
    print(f"{pid}/{processName}/{threadName} ---> Starting Main...")

    print("Ready For Takeoff!\n")
    print(f'{"Auto" if driver.auto else "Manual"} Mode',
          f'---> {driver.get_drive_state_label()}',
          f'---> Motors: {"On" if driver.motor_state else "Off"}')
    try:
        while driver.loop:
            if driver.auto:
                obj_dist = dist.distance()
                if (obj_dist > 0) and (obj_dist < stop_dist):
                    print(f'OBSTRUCTION DETECTED: {obj_dist:.2f}/{stop_dist}cm')
                    driver.motor_off()
                    while obj_dist < stop_dist:
                        obj_dist = dist.distance()
                    print("Continuing...")
                    driver.motor_on()
                else:
                    img = eye.get_thresh_img()
                    motor_pred, cert = auto_motor.drive_predict(img)
                    print(motor_pred[1])
                    auto_drive_handler(driver, motor_pred[1])
            else:
                img = eye.get_img()
                img_thresh = eye.get_thresh_img(img)
                img_horiz_stack = E.stack_images(1, ([img, img_thresh]))
                eye.show_img(img_horiz_stack, "Camera View", 1)

    except Exception as e:
        print(e)
    except KeyboardInterrupt as e:
        print(e)
        print("Shutting Down...")
    finally:
        if not stop_event.is_set():
            stop_event.set()
        t1_io.join()
        try:
            m.destroy()
        except Exception as e:
            print(e)
        try:
            controller.kill_device()
        except Exception as e:
            print(e)
        try:
            eye.destroy()
        except Exception as e:
            print(e)
        print('Done.')
