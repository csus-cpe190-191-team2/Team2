import ultrasonic_distance
import motor as mc
import RPi.GPIO as GPIO

def handle_distance(distance):
    mc.setup()
    motor = mc.MotorControl()
    #if Greater than 6 (6*2.54 = 15.24cm) inches go full speed (set_speed(3))
    if distance > 15.24:
        speed = motor.set_speed(3)
    #else if Less than 6 (6*2.54 = 15.24cm) inches but greater than 4 (4*2.54 = 10.16) inches (set_speed(2))
    elif distance < 15.24 and distance > 10.16:
        speed = motor.set_speed(2)
        print("Speed set to 2")
    #else if Less than 4 (4*2.54 = 10.16cm) inches but greater than 2 (2*2.54 = 5.08) inches (set_speed(1))
    elif distance < 10.16 and distance > 5.08:
        speed = motor.set_speed(1)
        print("Speed set to 1")
    #else Less than 2 (2*2.54 = 5.08cm) inches (set_speed(0))
    else:
        speed = motor.set_speed(0)
        print("Stop")

    #Return boolean value: if speed > 0 return True, else return False
    if max(motor.left_duty, motor.right_duty) > 0:
        return True
    else:
        return False
    mc.destroy()

if __name__ == '__main__':  # Program start from here
    while True:
        dist = ultrasonic_distance.distance()
        print(dist)
        handle_distance(dist)

    GPIO.cleanup()
