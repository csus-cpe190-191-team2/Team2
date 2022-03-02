import RPi.GPIO as GPIO
import math
from time import sleep

# GPIO Pins for Motor Driver Inputs
Motor1A = 16        # Green Wire 16
Motor2A = 18        # Blue Wire 18
MotorA_PWM = 24     # 22
Motor1B = 19        # Green Wire 19
Motor2B = 26        # Blue Wire 26
MotorB_PWM = 22     # 24
Stby = 15


# Reset ports used for GPIO
def destroy():
    GPIO.cleanup()


def setup():
    GPIO.setmode(GPIO.BOARD)  # GPIO Pin Numbering Scheme
    GPIO.setup(Motor1A, GPIO.OUT)
    GPIO.setup(Motor2A, GPIO.OUT)
    GPIO.setup(MotorA_PWM, GPIO.OUT)

    GPIO.setup(Motor1B, GPIO.OUT)
    GPIO.setup(Motor2B, GPIO.OUT)
    GPIO.setup(MotorB_PWM, GPIO.OUT)
    GPIO.setup(Stby, GPIO.OUT)


class MotorControl:
    def __init__(self):
        self.MAX_DUTY = 90   # Limit max motor speed to avoid damage
        self.left_duty = 0
        self.right_duty = 0
        setup()              # Setup GPIO
        # Initiate PWM
        self.left_motor = GPIO.PWM(MotorA_PWM, 100)
        self.right_motor = GPIO.PWM(MotorB_PWM, 100)
        self.left_motor.start(self.left_duty)
        self.right_motor.start(self.right_duty)
        # Set motor configuration to spin forwards
        GPIO.output(Motor1A, GPIO.HIGH)
        GPIO.output(Motor2A, GPIO.LOW)
        GPIO.output(Motor1B, GPIO.HIGH)
        GPIO.output(Motor2B, GPIO.LOW)
        # Keep motor controller in standby mode to start
        GPIO.output(Stby, GPIO.LOW)

    # Configure motors to move in a forward direction
    def forward(self):
        if not GPIO.input(Stby):    # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        # Sync motors
        avg_duty = max(self.left_duty, self.right_duty)
        self.left_duty = self.right_duty = avg_duty
        # Reduce motor speeds gradually to prepare for forward motion
        for i in range(avg_duty, -1, -1):
            self.left_motor.ChangeDutyCycle(i)
            self.right_motor.ChangeDutyCycle(i)
            sleep(0.01)  # Sleep 1ms

        # Set motor configuration to spin forwards
        GPIO.output(Motor1A, GPIO.HIGH)
        GPIO.output(Motor2A, GPIO.LOW)
        GPIO.output(Motor1B, GPIO.HIGH)
        GPIO.output(Motor2B, GPIO.LOW)

        # Increase motor speeds gradually to original speed before direction change
        for i in range(0, avg_duty+1, +1):
            self.left_motor.ChangeDutyCycle(i)
            self.right_motor.ChangeDutyCycle(i)
            sleep(0.01)  # Sleep 1ms

    # Configure motors to move in a backward direction
    def backward(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        # Normalize motor data, find average duty cycle and sync motors
        avg_duty = max(self.left_duty, self.right_duty)
        self.left_duty = self.right_duty = avg_duty

        # Reduce motor speeds gradually to prepare for forward motion
        for i in range(avg_duty, -1, -1):
            self.left_motor.ChangeDutyCycle(i)
            self.right_motor.ChangeDutyCycle(i)
            sleep(0.01)  # Sleep 1ms

        # Set motor configuration to spin backwards
        GPIO.output(Motor1A, GPIO.LOW)
        GPIO.output(Motor2A, GPIO.HIGH)
        GPIO.output(Motor1B, GPIO.LOW)
        GPIO.output(Motor2B, GPIO.HIGH)

        # Increase motor speeds gradually to original speed before direction change
        for i in range(0, avg_duty+1, +1):
            self.left_motor.ChangeDutyCycle(i)
            self.right_motor.ChangeDutyCycle(i)
            sleep(0.01)  # Sleep 1ms

    # Reduce left motor by half to turn left
    def turn_left(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        self.left_duty -= 40
        self.left_motor.ChangeDutyCycle(self.left_duty)

    # Reduce right motor by half to turn right
    def turn_right(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        self.right_duty -= 40
        self.right_motor.ChangeDutyCycle(self.right_duty)

    # ### NEEDS WORK: Increase speed of motor
    def speed_up(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        if (self.left_duty + 5) < self.MAX_DUTY:
            self.left_duty = math.floor(self.left_duty + 5)
        if (self.right_duty + 5) < self.MAX_DUTY:
            self.right_duty = math.floor(self.right_duty + 5)
        self.left_motor.ChangeDutyCycle(self.left_duty)
        self.right_motor.ChangeDutyCycle(self.right_duty)

    # ### NEEDS WORK: Decrease speed of motor
    def speed_down(self):
        # If in standby mode and at least one motor duty cycle is above zero, activate motor controller
        if not GPIO.input(Stby) and ((self.left_duty != 0) or (self.right_duty != 0)):
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        self.left_duty = math.floor(self.left_duty / 1.5)
        self.right_duty = math.floor(self.right_duty / 1.5)
        self.left_motor.ChangeDutyCycle(self.left_duty)
        self.right_motor.ChangeDutyCycle(self.right_duty)
        # If speed is zero place controller in standby mode to conserve power
        if (self.left_duty == 0) and (self.right_duty == 0):
            GPIO.output(Stby, GPIO.LOW)

    def stop(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        avg_duty = math.floor((self.left_duty + self.right_duty) / 2)

        # Reduce motor speeds gradually to prepare for forward motion
        for i in range(avg_duty, -1, -1):
            self.left_motor.ChangeDutyCycle(i)
            self.right_motor.ChangeDutyCycle(i)
            sleep(0.01)  # Sleep 1ms

        # If place motor controller in standby mode to conserve power
        GPIO.output(Stby, GPIO.LOW)
