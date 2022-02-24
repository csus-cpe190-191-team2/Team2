from evdev import InputDevice,categorize,ecodes

gamepad=InputDevice('/dev/input/event1')
#find out what event the device is first^^^

a=34
b=36
y=23
x=35
start=24
select=49
up=46
down=32
left=18
right=33
Ltrigger=37
Rtrigger=50

print(gamepad)

for event in gamepad.read_loop():
    if event.type==ecodes.EV_KEY:
        if event.value==1:
            if event.code==a:
                print('a')
            elif event.code==b:
                print('b')
            elif event.code==y:
                print('y')
            elif event.code==x:
                print('x')
            elif event.code==start:
                print('start')
            elif event.code==select:
                print('select')
            elif event.code==up:
                print('up')
            elif event.code==down:
                print('down')
            elif event.code==left:
                print('left')
            elif event.code==right:
                print('right')
            elif event.code==Ltrigger:
                print('Left Trigger')
            elif event.code==Rtrigger:
                print('Right Trigger')
