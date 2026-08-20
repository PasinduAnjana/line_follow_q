#!/usr/bin/env pybricks-micropython

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import ColorSensor, InfraredSensor
from pybricks.parameters import Port
from pybricks.tools import wait

ev3 = EV3Brick()
color_sensor = ColorSensor(Port.S1)

try:
    ir_sensor = InfraredSensor(Port.S4)
except Exception:
    ir_sensor = None

print("--- SENSOR CALIBRATION RUNNING ---")

while True:
    reflection = color_sensor.reflection()
    
    # Classify reflection according to main.py calibrated thresholds
    if reflection >= 25:
        surface = "WHITE TAPE"
    elif reflection >= 15:
        surface = "ON EDGE"
    elif reflection >= 7:
        surface = "DRIFT BLACK"
    else:
        surface = "LOST / BLACK"

    # Read IR distance if available
    ir_dist = ir_sensor.distance() if ir_sensor else 0

    # 📺 Render on EV3 Screen
    ev3.screen.clear()
    ev3.screen.draw_text(5, 5,  "--- CALIBRATE ---")
    ev3.screen.draw_text(5, 30, "Reflect: " + str(reflection) + "%")
    ev3.screen.draw_text(5, 50, "Surface: " + surface)
    if ir_sensor:
        ev3.screen.draw_text(5, 75, "IR Dist: " + str(ir_dist) + "cm")
        if ir_dist < 10:
            ev3.screen.draw_text(5, 95, "[!] OBSTACLE NEAR")

    # Terminal output
    print("Reflection: {:3d}% | Surface: {:12s} | IR Dist: {:3d}cm".format(
        reflection, surface, ir_dist
    ))

    wait(150)