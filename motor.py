import RPi.GPIO as GPIO

# GPIO Pins for Motor Driver Inputs
Motor1A = 16  # Green Wire 16
Motor2A = 18  # Blue Wire 18
MotorA_PWM = 24
Motor1B = 19  # Green Wire 19
Motor2B = 26  # Blue Wire 26
MotorB_PWM = 22
Stby = 15


# Reset ports used for GPIO
def destroy():
    GPIO.cleanup()


# GPIO setup
def setup():
    GPIO.setmode(GPIO.BOARD)
    ##
    GPIO.setup(Stby, GPIO.OUT)
    ##
    GPIO.setup(Motor1A, GPIO.OUT)
    GPIO.setup(Motor2A, GPIO.OUT)
    GPIO.setup(MotorA_PWM, GPIO.OUT)
    ##
    GPIO.setup(Motor1B, GPIO.OUT)
    GPIO.setup(Motor2B, GPIO.OUT)
    GPIO.setup(MotorB_PWM, GPIO.OUT)


# Front facing motor configuration
def front():
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)


class MotorControl:
    def __init__(self):
        self.MAX_DUTY = 100  # Limit max motor speed
        self.MED_DUTY = 85  # Medium speed
        self.MIN_DUTY = 70  # Low speed
        self.current_duty = 0
        self.motor_state = False
        self.drive_state = 0
        self.auto = False  # Toggles autonomous mode
        self.loop = True  # Toggles main loop
        setup()  # Setup GPIO

        # Keep motor controller in standby mode to start
        GPIO.output(Stby, GPIO.LOW)

        # Set motor configuration to spin forwards
        front()

        # Initiate PWM
        self.left_motor = GPIO.PWM(MotorA_PWM, 100)
        self.right_motor = GPIO.PWM(MotorB_PWM, 100)
        self.left_motor.start(0)  # Start PWM
        self.right_motor.start(0)  # Start PWM

    def get_drive_state_label(self):
        states = ["stopped", "forward", "backward", "left",
                  "right", "Rright", "Rleft"]
        return states[self.drive_state]

    def toggle_auto(self):
        if self.auto:
            self.auto = False
        else:
            self.auto = True

    def toggle_motor(self):
        # stby: Allow H-bridges to work when high
        if self.motor_state:
            GPIO.output(Stby, GPIO.LOW)
            self.motor_state = False
        else:
            GPIO.output(Stby, GPIO.HIGH)
            self.motor_state = True

    def off_duty(self):
        if self.motor_state:
            self.left_motor.ChangeDutyCycle(0)
            self.right_motor.ChangeDutyCycle(0)

    # Configure motors to move in a forward direction
    def forward(self):
        if self.motor_state:
            front()
            self.drive_state = 1
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    # Configure motors to move in a backward direction
    def backward(self):
        if self.motor_state:
            GPIO.output(Motor1A, GPIO.LOW)
            GPIO.output(Motor2A, GPIO.HIGH)
            GPIO.output(Motor1B, GPIO.LOW)
            GPIO.output(Motor2B, GPIO.HIGH)
            self.drive_state = 2
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    # Reduce left motor speed to turn left
    def turn_left(self):
        if self.motor_state:
            front()
            self.drive_state = 3
            self.left_motor.ChangeDutyCycle(self.MED_DUTY / 2)
            self.right_motor.ChangeDutyCycle(self.MAX_DUTY)

    # Reduce right motor speed to turn right
    def turn_right(self):
        if self.motor_state:
            front()
            self.drive_state = 4
            self.left_motor.ChangeDutyCycle(self.MAX_DUTY)
            self.right_motor.ChangeDutyCycle(self.MED_DUTY / 2)

    # Increase speed of motor
    def speed_up(self):
        if self.motor_state:
            if self.current_duty == 0:
                self.current_duty = self.MIN_DUTY
            elif self.current_duty == self.MIN_DUTY:
                self.current_duty = self.MED_DUTY
            else:
                self.current_duty = self.MAX_DUTY

            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    # Decrease speed of motor
    def speed_down(self):
        if self.motor_state:
            if self.current_duty == self.MAX_DUTY:
                self.current_duty = self.MED_DUTY
            elif self.current_duty == self.MED_DUTY:
                self.current_duty = self.MIN_DUTY
            else:
                self.current_duty = 0

            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def rotate_right(self):
        if self.motor_state:
            GPIO.output(Motor1A, GPIO.LOW)
            GPIO.output(Motor2A, GPIO.HIGH)
            GPIO.output(Motor1B, GPIO.HIGH)
            GPIO.output(Motor2B, GPIO.LOW)
            self.drive_state = 5
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def rotate_left(self):
        if self.motor_state:
            GPIO.output(Motor1A, GPIO.HIGH)
            GPIO.output(Motor2A, GPIO.LOW)
            GPIO.output(Motor1B, GPIO.LOW)
            GPIO.output(Motor2B, GPIO.HIGH)
            self.drive_state = 6
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    # Start moving forward at a default speed
    def default_duty(self):
        # Turn on motor controller
        if not self.motor_state:  # If motor is off, turn on
            self.toggle_motor()

        # Set to best default speed
        self.current_duty = self.MIN_DUTY

        # Set the duty cycle
        self.left_motor.ChangeDutyCycle(self.current_duty)
        self.right_motor.ChangeDutyCycle(self.current_duty)

    # ### MAY BE DEPRECIATED ###
    # Used for automation to set desired speed: Expects int in range: 0-3
    def set_speed(self, speed=0):
        if speed == 3:
            self.current_duty = self.MAX_DUTY
        elif speed == 2:
            self.current_duty = self.MED_DUTY
        elif speed == 1:
            self.current_duty = self.MIN_DUTY
        else:
            self.current_duty = 0

        # If in standby mode and at least one motor duty cycle is above zero, activate motor controller
        if not self.motor_state:  # Motors disabled when not active
            # Stby: Allow H-bridges to work when high
            # (has a pull down resistor must be actively pulled HIGH)
            self.toggle_motor()

        self.left_motor.ChangeDutyCycle(self.current_duty)
        self.right_motor.ChangeDutyCycle(self.current_duty)
