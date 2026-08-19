#!/usr/bin/env pybricks-micropython

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor, InfraredSensor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait
import random
import json

# =====================================================================
# 🔌 1. HARDWARE INITIALIZATION
# =====================================================================
ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.D)
color_sensor = ColorSensor(Port.S1)
ir_sensor = InfraredSensor(Port.S4)

# =====================================================================
# 🧠 2. HYPERPARAMETERS & MODE SETTING
# =====================================================================
# Set to False for DEMO / EVALUATION (uses the loaded Q-table without changes)
# Set to True for TRAINING (learns and updates the Q-table file)
TRAINING_MODE = False  

ALPHA = 0.25          # Learning rate
GAMMA = 0.85          # Discount factor for future rewards
EPSILON = 0.40 if TRAINING_MODE else 0.0  # 0.0 for pure exploitation (Demo mode)
MIN_EPSILON = 0.02    # Minimum exploration
DECAY_RATE = 0.997    # Slow decay for thorough learning

# Actions (Assignment required: Forward, Reverse, Left, Right)
FORWARD, REVERSE, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["FWD", "REV", "LEFT", "RIGHT"]

# States based on calibrated reflection (White: ~30, Edge: ~20, Black: ~4)
ON_EDGE      = 0   # Perfect tracking on boundary (reflection 15 to 24) -> Learns FORWARD
TOO_WHITE    = 1   # Deep on white tape (reflection >= 25) -> Learns turn towards edge
TOO_BLACK    = 2   # Drifting into black background (reflection 7 to 14) -> Learns turn towards edge
LOST_OFFLINE = 3   # Deep in black / completely lost (reflection < 7) -> Learns REVERSE / Turn!
NUM_STATES   = 4
NUM_ACTIONS  = 4

# Loop Direction Setting: "CLOCKWISE" or "ANTICLOCKWISE"
# On the big loop:
# - CLOCKWISE: Inner edge has black on the RIGHT, white tape on the LEFT.
# - ANTICLOCKWISE: Inner edge has black on the LEFT, white tape on the RIGHT.
DIRECTION = "CLOCKWISE"

Q_TABLE_FILE = "q_table_v7.json"

# =====================================================================
# 💾 3. SAVE & LOAD Q-TABLE
# =====================================================================
def save_q_table(q_table, filename=Q_TABLE_FILE):
    try:
        with open(filename, "w") as f:
            json.dump(q_table, f)
        print("💾 Q-Table saved successfully!")
    except Exception as e:
        print("Failed to save Q-table:", e)

def load_q_table(filename=Q_TABLE_FILE):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
            # Validate dimensions
            if len(data) == NUM_STATES and len(data[0]) == NUM_ACTIONS:
                print("📂 Loaded saved Q-Table from", filename)
                return data
    except OSError:
        pass

    print("🆕 Starting with fresh Q-table!")
    return [[0.0 for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES)]

q_table = load_q_table()

# =====================================================================
# 🎯 4. SENSING & STATE DETERMINATION (CALIBRATED)
# =====================================================================
def get_state():
    """
    Calibrated based on:
    - White path: ~30 (>= 25)
    - Edge boundary: ~20 (15..24)
    - Drifting black: ~10 (7..14)
    - Deep black (Lost): ~4 (< 7)
    """
    reflection = color_sensor.reflection()
    if reflection >= 25:
        return TOO_WHITE
    elif reflection >= 15:
        return ON_EDGE
    elif reflection >= 7:
        return TOO_BLACK
    else:
        return LOST_OFFLINE

# =====================================================================
# 🕹️ 5. ACTION SELECTION & EXECUTION
# =====================================================================
def select_action(state):
    """Epsilon-greedy action selection."""
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        # Exploit: pick action with maximum Q-value for current state
        state_q = q_table[state]
        max_q = max(state_q)
        # Random tie-breaker among best actions
        best_actions = [i for i, q in enumerate(state_q) if q == max_q]
        return random.choice(best_actions)

def execute_action(action):
    """
    Stronger pivot turns to easily wrap around 90-degree rectangular corners.
    """
    speed_fwd  = 110
    turn_outer = 150
    turn_inner = -70    # Strong counter-rotation to turn 90-deg corners fast
    rev_speed  = 90

    if action == FORWARD:
        left_motor.run(speed_fwd)
        right_motor.run(speed_fwd)
    elif action == REVERSE:
        left_motor.run(-rev_speed)
        right_motor.run(-rev_speed)
    elif action == LEFT:
        left_motor.run(turn_inner)
        right_motor.run(turn_outer)
    elif action == RIGHT:
        left_motor.run(turn_outer)
        right_motor.run(turn_inner)

    wait(100)  # 100ms duration ensures adequate degrees per turn step

# =====================================================================
# 🏆 6. REWARD FUNCTION (DIRECTION-AWARE INNER EDGE TRACKING)
# =====================================================================
def get_reward(prev_state, action, new_state):
    """
    Enforces inner-edge tracking so the robot NEVER crosses onto the
    outer loop or gets stuck in reverse oscillations.
    """
    # Define correct turn directions based on track orientation
    if DIRECTION == "CLOCKWISE":
        turn_towards_tape = LEFT   # Tape is to the left
        turn_towards_edge = RIGHT  # Center/edge is to the right
    else: # ANTICLOCKWISE
        turn_towards_tape = RIGHT  # Tape is to the right
        turn_towards_edge = LEFT   # Center/edge is to the left

    # 1. On edge -> Moving forward is ideal
    if prev_state == ON_EDGE:
        if action == FORWARD:
            return 12.0
        elif action == REVERSE:
            return -8.0
        else:
            return -2.0  # Mild penalty for turning when already on edge

    # 2. Too far on white tape -> MUST turn towards inner edge (NOT across to outer loop!)
    elif prev_state == TOO_WHITE:
        if action == turn_towards_edge:
            return 15.0 if new_state == ON_EDGE else 6.0
        elif action == turn_towards_tape:
            return -12.0 # HEAVILY penalize crossing over the tape to outer edge!
        elif action == FORWARD:
            return 1.0
        else:
            return -6.0

    # 3. Drifting into black center -> MUST turn towards white tape
    elif prev_state == TOO_BLACK:
        if action == turn_towards_tape:
            return 15.0 if new_state in (ON_EDGE, TOO_WHITE) else 6.0
        elif action == turn_towards_edge:
            return -12.0 # HEAVILY penalize steering further into black!
        elif action == FORWARD:
            return -4.0
        else:
            return -5.0

    # 4. Completely lost off track (< 6 reflection)
    elif prev_state == LOST_OFFLINE:
        if action == turn_towards_tape and new_state in (ON_EDGE, TOO_WHITE, TOO_BLACK):
            return 16.0
        elif action == REVERSE:
            if new_state in (ON_EDGE, TOO_WHITE, TOO_BLACK):
                return 8.0
            return 3.0
        elif action == FORWARD:
            return -15.0
        else:
            return -6.0

    return -5.0

# =====================================================================
# 🚧 7. NON-RL OBSTACLE AVOIDANCE & PATH FINDING
# =====================================================================
def avoid_obstacle():
    """
    Non-RL bypass maneuver when an obstacle is detected:
    1. Stop and alert (beep)
    2. Turn into the open inner center area (away from track and obstacle)
    3. Drive forward along the bypass flank past the obstacle
    4. Turn back to face the track
    """
    print("🚧 Obstacle detected! Starting bypass maneuver...")
    ev3.speaker.beep(frequency=700, duration=150)
    left_motor.stop(Stop.BRAKE)
    right_motor.stop(Stop.BRAKE)
    wait(200)

    # In CLOCKWISE, open center is to the RIGHT -> Turn Right first
    # In ANTICLOCKWISE, open center is to the LEFT -> Turn Left first
    if DIRECTION == "CLOCKWISE":
        # Turn Right ~75 deg
        left_motor.run_angle(180, 160)
        right_motor.run_angle(180, -160)
        wait(200)

        # Drive forward past obstacle
        left_motor.run_angle(180, 260, wait=False)
        right_motor.run_angle(180, 260, wait=True)
        wait(200)

        # Turn Left ~75 deg to aim back toward the line
        left_motor.run_angle(180, -170)
        right_motor.run_angle(180, 170)
        wait(200)
    else: # ANTICLOCKWISE
        # Turn Left ~75 deg
        left_motor.run_angle(180, -160)
        right_motor.run_angle(180, 160)
        wait(200)

        # Drive forward past obstacle
        left_motor.run_angle(180, 260, wait=False)
        right_motor.run_angle(180, 260, wait=True)
        wait(200)

        # Turn Right ~75 deg to aim back toward the line
        left_motor.run_angle(180, 170)
        right_motor.run_angle(180, -170)
        wait(200)

def find_path():
    """
    Non-RL sweep routine to re-acquire the path after bypass or losing line:
    Drives forward/sweeps until the line's reflection threshold is detected.
    """
    print("🔍 Searching for line path...")
    # Drive forward diagonally towards the line
    left_motor.run(110)
    right_motor.run(110)

    timeout = 0
    while color_sensor.reflection() < 16 and timeout < 40:
        wait(50)
        timeout += 1

    # If not found directly, sweep in an arc to find the line
    if color_sensor.reflection() < 16:
        left_motor.run(-80)
        right_motor.run(80)
        while color_sensor.reflection() < 16:
            wait(50)

    left_motor.stop(Stop.BRAKE)
    right_motor.stop(Stop.BRAKE)
    print("✅ Path found! Resuming RL control.")
    ev3.speaker.beep(frequency=1200, duration=100)
    wait(200)

# =====================================================================
# 🚀 8. MAIN EXECUTION / EVALUATION LOOP
# =====================================================================
step_count = 0
status_str = "Running Demo" if not TRAINING_MODE else "Training"
ev3.speaker.say(status_str)
print("🚀 Mode:", status_str, "| Direction:", DIRECTION)

while True:
    # 1. Non-RL Obstacle detection check (IR distance < 15cm)
    if ir_sensor.distance() < 15:
        avoid_obstacle()
        find_path()
        continue

    # 2. Sense current state
    state = get_state()

    # 3. Select action (Epsilon-Greedy / Pure Exploitation)
    action = select_action(state)

    # 4. Execute selected action
    execute_action(action)

    # 5. Observe new state & calculate reward
    new_state = get_state()
    reward = get_reward(state, action, new_state)

    # 6. Q-Learning Update (Only active during TRAINING_MODE)
    if TRAINING_MODE:
        max_future_q = max(q_table[new_state])
        current_q = q_table[state][action]
        q_table[state][action] = current_q + ALPHA * (reward + (GAMMA * max_future_q) - current_q)
        EPSILON = max(MIN_EPSILON, EPSILON * DECAY_RATE)

    step_count += 1

    # 7. On-screen EV3 Dashboard
    if step_count % 5 == 0:
        ev3.screen.clear()
        ev3.screen.draw_text(5, 5,  "Mode: " + ("DEMO" if not TRAINING_MODE else "TRAIN"))
        ev3.screen.draw_text(5, 25, "Step: " + str(step_count))
        ev3.screen.draw_text(5, 45, "State:" + str(state) + " Act:" + ACTION_NAMES[action])
        ev3.screen.draw_text(5, 65, "Rwd:  " + str(round(reward, 1)))

    # 8. Periodically save Q-table during training
    if TRAINING_MODE and step_count % 25 == 0:
        save_q_table(q_table)

    print("Step: {:4d} | State: {:d} | Action: {:5s} | Reward: {:5.1f} | Eps: {:.3f}".format(
        step_count, state, ACTION_NAMES[action], reward, EPSILON
    ))