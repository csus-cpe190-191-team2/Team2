import RPi.GPIO as GPIO
from time import sleep

# GPIO Pins for Motor Driver Inputs
Motor1A = 16        # Green Wire 16
Motor2A = 18        # Blue Wire 18
MotorA_PWM = 24
Motor1B = 19        # Green Wire 19
Motor2B = 26        # Blue Wire 26
MotorB_PWM = 22
Stby = 15


# Reset ports used for GPIO
def destroy():
    GPIO.cleanup()


# GPIO setup
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
        self.MAX_DUTY = 100     # Limit max motor speed
        self.MED_DUTY = 85      # Medium speed
        self.MIN_DUTY = 65      # Low speed
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
        max_duty = max(self.left_duty, self.right_duty)
        self.left_duty = self.right_duty = max_duty

        # If removing in reverse reduce speed to prepare to reverse motor direction
        if GPIO.input(Motor2A) and GPIO.input(Motor2B):
            # Reduce motor speeds gradually to prepare for forward motion
            for i in range(max_duty, -1, -1):
                self.left_motor.ChangeDutyCycle(i)
                self.right_motor.ChangeDutyCycle(i)
                sleep(0.01)  # Sleep 1ms

        # Set motor configuration to spin forwards
        GPIO.output(Motor1A, GPIO.HIGH)
        GPIO.output(Motor2A, GPIO.LOW)
        GPIO.output(Motor1B, GPIO.HIGH)
        GPIO.output(Motor2B, GPIO.LOW)

        self.left_motor.ChangeDutyCycle(max_duty)
        self.right_motor.ChangeDutyCycle(max_duty)

    # Configure motors to move in a backward direction
    def backward(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        # Normalize motor data, find average duty cycle and sync motors
        max_duty = max(self.left_duty, self.right_duty)
        self.left_duty = self.right_duty = max_duty

        if GPIO.input(Motor1A) and GPIO.input(Motor1B):
            # Reduce motor speeds gradually to prepare for forward motion
            for i in range(max_duty, -1, -1):
                self.left_motor.ChangeDutyCycle(i)
                self.right_motor.ChangeDutyCycle(i)
                sleep(0.01)  # Sleep 1ms

        # Set motor configuration to spin backwards
        GPIO.output(Motor1A, GPIO.LOW)
        GPIO.output(Motor2A, GPIO.HIGH)
        GPIO.output(Motor1B, GPIO.LOW)
        GPIO.output(Motor2B, GPIO.HIGH)

        self.left_motor.ChangeDutyCycle(max_duty)
        self.right_motor.ChangeDutyCycle(max_duty)

    # Reduce left motor speed to turn left
    def turn_left(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        self.left_duty = self.right_duty = max(self.left_duty, self.right_duty)

        # Rotate left in place
        if self.left_duty == 0 and self.right_duty == 0:
            GPIO.output(Motor1A, GPIO.HIGH)
            GPIO.output(Motor2A, GPIO.LOW)
            GPIO.output(Motor1B, GPIO.LOW)
            GPIO.output(Motor2B, GPIO.HIGH)
            self.left_duty = self.right_duty = self.MED_DUTY
        # Turn left
        else:
            self.left_duty = self.MIN_DUTY
            self.right_duty = self.MAX_DUTY

        self.left_motor.ChangeDutyCycle(self.left_duty)
        self.right_motor.ChangeDutyCycle(self.right_duty)

    # Reduce right motor speed to turn right
    def turn_right(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)
            GPIO.output(Motor2A, GPIO.HIGH)

        self.left_duty = self.right_duty = max(self.left_duty, self.right_duty)

        # Rotate right in place
        if self.left_duty == 0 and self.right_duty == 0:
            GPIO.output(Motor1A, GPIO.LOW)
            GPIO.output(Motor2A, GPIO.HIGH)
            GPIO.output(Motor1B, GPIO.HIGH)
            GPIO.output(Motor2B, GPIO.LOW)
            self.left_duty = self.right_duty = self.MED_DUTY
        # Turn Left
        else:
            self.right_duty = self.MIN_DUTY
            self.left_duty = self.MAX_DUTY

        self.right_motor.ChangeDutyCycle(self.right_duty)
        self.left_motor.ChangeDutyCycle(self.left_duty)

    # Increase speed of motor
    def speed_up(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        max_duty = max(self.left_duty, self.right_duty)

        if max_duty == 0:
            self.left_duty = self.MIN_DUTY
            self.right_duty = self.MIN_DUTY
        elif max_duty == self.MIN_DUTY:
            self.left_duty = self.MED_DUTY
            self.right_duty = self.MED_DUTY
        else:
            self.left_duty = self.MAX_DUTY
            self.right_duty = self.MAX_DUTY

        self.left_motor.ChangeDutyCycle(self.left_duty)
        self.right_motor.ChangeDutyCycle(self.right_duty)

    # Decrease speed of motor
    def speed_down(self):
        # If in standby mode and at least one motor duty cycle is above zero, activate motor controller
        if not GPIO.input(Stby) and ((self.left_duty != 0) or (self.right_duty != 0)):
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        max_duty = max(self.left_duty, self.right_duty)

        if max_duty == self.MAX_DUTY:
            self.left_duty = self.MED_DUTY
            self.right_duty = self.MED_DUTY
        elif max_duty == self.MED_DUTY:
            self.left_duty = self.MIN_DUTY
            self.right_duty = self.MIN_DUTY
        else:
            self.left_duty = 0
            self.right_duty = 0

        self.left_motor.ChangeDutyCycle(self.left_duty)
        self.right_motor.ChangeDutyCycle(self.right_duty)

        # If speed is zero place controller in standby mode to conserve power
        if (self.left_duty == 0) and (self.right_duty == 0):
            GPIO.output(Stby, GPIO.LOW)

    # Used for automation to set desired speed: Expects int in range: 0-3
    def set_speed(self, speed=0):
        if speed == 3:
            self.left_duty = self.MAX_DUTY
            self.right_duty = self.MAX_DUTY
        elif speed == 2:
            self.left_duty = self.MED_DUTY
            self.right_duty = self.MED_DUTY
        elif speed == 1:
            self.left_duty = self.MIN_DUTY
            self.right_duty = self.MIN_DUTY
        else:
            self.left_duty = 0
            self.right_duty = 0

        # If in standby mode and at least one motor duty cycle is above zero, activate motor controller
        if not GPIO.input(Stby) and ((self.left_duty != 0) or (self.right_duty != 0)):
            GPIO.output(Stby, GPIO.HIGH)
        else:
            GPIO.output(Stby, GPIO.LOW)

        self.left_motor.ChangeDutyCycle(self.left_duty)
        self.right_motor.ChangeDutyCycle(self.right_duty)

    # Stop motors
    def stop(self):
        if not GPIO.input(Stby):  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            GPIO.output(Stby, GPIO.HIGH)

        max_duty = min(self.left_duty, self.right_duty)

        # Reduce motor speed gradually
        for i in range(max_duty, -1, -1):
            self.left_motor.ChangeDutyCycle(i)
            self.right_motor.ChangeDutyCycle(i)
            sleep(0.01)  # Sleep 1ms

        self.left_duty = self.right_duty = 0

        # If place motor controller in standby mode to conserve power
        GPIO.output(Stby, GPIO.LOW)
