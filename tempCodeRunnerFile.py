#!/usr/bin/env pybricks-micropython

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor, InfraredSensor
from pybricks.parameters import Port
from pybricks.tools import wait
import random
import json

# Initialize Hardware 🔌
ev3 = EV3Brick()
left_motor = Motor(Port.D)
right_motor = Motor(Port.B)     # Adjust port if right motor is on B/C
color_sensor = ColorSensor(Port.S4)
ir_sensor = InfraredSensor(Port.S1)

# Hyperparameters
ALPHA = 0.2     # Learning rate 🎓
GAMMA = 0.8     # Discount factor 🔮
EPSILON = 0.3  # Set to 0.3 for training, 0.0 for demo day 🎲

# Definitions
FORWARD, REVERSE, LEFT, RIGHT = 0, 1, 2, 3
ON_LINE, OFF_LEFT, OFF_RIGHT = 0, 1, 2

# --- SAVE & LOAD FUNCTIONS 💾 ---

def save_q_table(q_table, filename="q_table.json"):
    try:
        with open(filename, "w") as f:
            json.dump(q_table, f)
        print("💾 Q-Table saved to EV3!")
    except Exception as e:
        print("Failed to save Q-table:", e)

def load_q_table(filename="q_table.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            print("📂 Loaded saved Q-Table!")
            return data
    except OSError:
        print("🆕 No saved Q-table found. Starting fresh!")
        return [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]

q_table = load_q_table()

def get_state():
    """ CALIBRATED STATE BOUNDARIES BASED ON YOUR SENSOR 🎯 """
    reflection = color_sensor.reflection()
    if reflection >= 22:          # White path (25-30)
        return ON_LINE
    elif reflection >= 8:         # Edge boundary (10-20)
        return OFF_LEFT
    else:                         # Black background (2-4)
        return OFF_RIGHT

def select_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, 3)
    else:
        return q_table[state].index(max(q_table[state]))

def execute_action(action):
    speed = 150
    if action == FORWARD:
        left_motor.run(speed)
        right_motor.run(speed)
    elif action == REVERSE:
        left_motor.run(-speed)
        right_motor.run(-speed)
    elif action == LEFT:
        left_motor.run(-speed)
        right_motor.run(speed)
    elif action == RIGHT:
        left_motor.run(speed)
        right_motor.run(-speed)
    
    wait(200)
    left_motor.stop()
    right_motor.stop()

def get_reward(state, action):
    if state == ON_LINE and action == FORWARD:
        return 10.0   # Reward for driving straight on line
    elif state == OFF_LEFT and action == RIGHT:
        return 5.0    # Reward for turning right back onto track
    elif state == OFF_RIGHT and action == LEFT:
        return 5.0    # Reward for turning left back onto track
    elif action == REVERSE:
        return -5.0   # Penalty for unnecessary backing up
    else:
        return -10.0  # Penalty for driving off line

def avoid_obstacle():
    """ Hardcoded maneuver around obstacle without RL """
    ev3.speaker.beep()
    left_motor.run_angle(200, 180)
    right_motor.run_angle(200, -180)
    wait(500)
    left_motor.run_angle(200, 300)
    right_motor.run_angle(200, 300)
    wait(500)

def find_path():
    """ Hardcoded sweep to re-locate white track """
    while color_sensor.reflection() < 15:
        left_motor.run(100)
        right_motor.run(-100)
        wait(100)
    left_motor.stop()
    right_motor.stop()

step_count = 0
ev3.speaker.say("Ready to train")

while True:
    # Obstacle Check (IR distance threshold < 15%)
    if ir_sensor.distance() < 15:
        avoid_obstacle()
        find_path()
        continue

    state = get_state()
    action = select_action(state)
    execute_action(action)
    
    new_state = get_state()
    reward = get_reward(state, action)

    # Q-Learning Update Equation 🧠
    max_future_q = max(q_table[new_state])
    current_q = q_table[state][action]
    
    q_table[state][action] = current_q + ALPHA * (reward + (GAMMA * max_future_q) - current_q)

    step_count += 1
    if step_count % 20 == 0:
        save_q_table(q_table)

    print("Step:", step_count, "| State:", state, "| Q-Table:", q_table)