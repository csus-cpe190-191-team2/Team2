import RPi.GPIO as GPIO
from time import sleep

# Pins for Motor Driver Inputs 
Motor1A = 18
Motor2A = 16
MotorA_PWM = 22
Motor1B = 19
Motor2B = 26
MotorB_PWM = 24
Stby = 15


def setup():
    GPIO.setmode(GPIO.BOARD)  # GPIO Numbering
    GPIO.setup(Motor1A, GPIO.OUT)  # All pins as Outputs
    GPIO.setup(Motor2A, GPIO.OUT)
    GPIO.setup(MotorA_PWM, GPIO.OUT)

    GPIO.setup(Motor1B, GPIO.OUT)
    GPIO.setup(Motor2B, GPIO.OUT)
    GPIO.setup(MotorB_PWM, GPIO.OUT)
    GPIO.setup(Stby, GPIO.OUT)


def loop():
    p_a = GPIO.PWM(MotorA_PWM, 100)
    p_b = GPIO.PWM(MotorB_PWM, 100)
    p_a.start(0)
    p_b.start(0)

    # Going forwards (one motor for now)
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.LOW)
    GPIO.output(Stby, GPIO.HIGH)
    print("Forward")

    p_a.ChangeDutyCycle(85)
    p_b.ChangeDutyCycle(85)
    print("85%")
    sleep(3)

    # Going backwards (one motor for now)
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)
    print("Reverse")

    p_a.ChangeDutyCycle(85)
    p_b.ChangeDutyCycle(85)
    print("85%")
    sleep(3)

    # Stop
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Stby, GPIO.LOW)


def destroy():
    GPIO.cleanup()


if __name__ == '__main__':  # Program start from here
    setup()
    try:
        print("loop")
        loop()
    except KeyboardInterrupt:
        print("destroy")
        destroy()
    finally:
        print("clean up")
        GPIO.cleanup()  # clean all gpio
