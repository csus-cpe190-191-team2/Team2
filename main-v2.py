import device_finder as df
import motor2 as m
from time import sleep

if __name__ == '__main__':
    print("Configuring Controller...")
    inputDev = df.Controller(0)
    inputDev.listen()
    while inputDev.loop:
        print("looping...")
        sleep(0.5)
    inputDev.kill_device()
    quit()