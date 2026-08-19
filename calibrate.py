#!/usr/bin/env pybricks-micropython

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import ColorSensor
from pybricks.parameters import Port
from pybricks.tools import wait

ev3 = EV3Brick()
color_sensor = ColorSensor(Port.S1)

print("--- COLOR SENSOR CALIBRATION ---")

while True:
    reflection = color_sensor.reflection()
    print("Reflection Value:", reflection)
    wait(300)