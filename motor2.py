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

def test():
    front()
    left_motor = GPIO.PWN(MotorA_PWM, 100)
    right_motor = GPIO.PWN(MotorB_PWM, 100)
    left_motor.start(25)
    right_motor.start(25)
    on()
    sleep(3)
    off()
    back()
    left_motor.ChangeDutyCycle(50)
    right_motor.ChangeDutyCycle(50)
    on()
    sleep(2)
    off()
    destroy()

def destroy():
    GPIO.cleanup()

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


def left():
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.HIGH)


def right():
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)


def front():
    GPIO.output(Motor1A, GPIO.HIGH)
    GPIO.output(Motor2A, GPIO.LOW)
    GPIO.output(Motor1B, GPIO.HIGH)
    GPIO.output(Motor2B, GPIO.LOW)


def back():
    GPIO.output(Motor1A, GPIO.LOW)
    GPIO.output(Motor2A, GPIO.HIGH)
    GPIO.output(Motor1B, GPIO.LOW)
    GPIO.output(Motor2B, GPIO.HIGH)

def on():
    GPIO.output(Stby, GPIO.HIGH)

def off():
    GPIO.output(Stby, GPIO.LOW)

class MotorControl:
    def __init__(self):
        self.MAX_DUTY = 100
        self.MED_DUTY = 65
        self.MIN_DUTY = 35
        self.ZERO_DUTY = 0
        self.current_duty = 0
        #self.left_duty = 0
        #self.right_duty = 0
        self.toggle = False
        #
        setup()
        #
        self.left_motor = GPIO.PWN(MotorA_PWM, 100) #test the second parameter
        self.right_motor = GPIO.PWN(MotorB_PWM, 100) ###
        #
        front()
        #
        self.left_motor.start(0)
        self.right_motor.start(0)
        #
        off()

    def toggle_state(self):
        if self.toggle:
            off()
            #self.left_duty = self.right_duty = self.ZERO_DUTY
            self.off_duty()
            self.current_duty = self.ZERO_DUTY
            self.toggle = False
        else:
            on()
            self.toggle = True

    def forward(self):
        if self.toggle:
            front()
            # Sync motors
            # max_duty = max(self.left_duty, self.right_duty)
            # self.left_duty = self.right_duty = max_duty
            #
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def turn_right(self, offset):
        if self.toggle:
            front()
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty-offset)

    def turn_left(self, offset):
        if self.toggle:
            front()
            self.left_motor.ChangeDutyCycle(self.current_duty-offset)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def backward(self):
        if self.toggle:
            back()
            # Sync motors
            # max_duty = max(self.left_duty, self.right_duty)
            # self.left_duty = self.right_duty = max_duty
            #
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def rotate_right(self):
        if self.toggle:
            right() ##check configuration!!!!!!!!!
            # Sync motors
            # max_duty = max(self.left_duty, self.right_duty)
            # self.left_duty = self.right_duty = max_duty
            #
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def rotate_left(self):
        if self.toggle:
            left()  ##check configuration!!!!!!!!!
            # Sync motors
            # max_duty = max(self.left_duty, self.right_duty)
            # self.left_duty = self.right_duty = max_duty
            #
            self.left_motor.ChangeDutyCycle(self.current_duty)
            self.right_motor.ChangeDutyCycle(self.current_duty)

    def speed_up(self, delta=10):
        if self.toggle:
            if not ((self.current_duty + delta) > 100):
               self.current_duty = self.current_duty + delta

    def speed_down(self, delta=10):
        if self.toggle:
            if not ((self.current_duty - delta) < 0):
               self.current_duty = self.current_duty - delta

    def default_duty(self):
        if self.toggle:
            self.current_duty = self.MIN_DUTY

    def off_duty(self):
        if self.toggle:
            self.left_motor.ChangeDutyCycle(self.ZERO_DUTY)
            self.right_motor.ChangeDutyCycle(self.ZERO_DUTY)

setup()
test()
destroy()