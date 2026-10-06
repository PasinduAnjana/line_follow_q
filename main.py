#!/usr/bin/env pybricks-micropython

from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor, InfraredSensor
from pybricks.parameters import Port, Stop
from pybricks.tools import wait
import random
import json
try:
    import uos as os
except ImportError:
    import os

# =====================================================================
# 🔌 1. HARDWARE INITIALIZATION
# =====================================================================
ev3 = EV3Brick()
left_motor = Motor(Port.C)
right_motor = Motor(Port.B)
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
EPSILON = 0.35 if TRAINING_MODE else 0.0  # Controlled exploration rate
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

# Obstacle Detection Distance (cm)
OBSTACLE_THRESHOLD_CM = 10

Q_TABLE_FILE = "q_table_v10.json"

# =====================================================================
# 💾 3. ATOMIC SAFE SAVE & LOAD Q-TABLE
# =====================================================================
def save_q_table(q_table, filename=Q_TABLE_FILE):
    """
    Atomic save: Writes to a .tmp file first, then renames.
    Prevents file corruption even if interrupted mid-save.
    """
    temp_file = filename + ".tmp"
    try:
        with open(temp_file, "w") as f:
            json.dump(q_table, f)
        os.rename(temp_file, filename)
        print("💾 Q-Table safely saved!")
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
    except Exception as e:
        print("⚠️ Q-Table file not ready or corrupt ({}). Starting fresh.".format(e))

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
    Motor control optimized for sharp 90-degree corners and smooth straights.
    """
    speed_fwd  = 100
    turn_outer = 160
    turn_inner = -90    # Fast counter-rotation for sharp 90-deg corner transitions
    rev_speed  = 85

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

    wait(85)  # Responsive 85ms sampling for instant corner detection

# =====================================================================
# 🏆 6. PURE STATE-TRANSITION REWARD FUNCTION (NO REWARD HACKING)
# =====================================================================
def get_reward(prev_state, action, new_state):
    """
    Pure transition-based reward R(s, a, s'):
    - Strongly rewards FORWARD on the edge.
    - Heavily penalizes FORWARD in TOO_WHITE/TOO_BLACK (prevents overshooting corners).
    - Only rewards corrective TURNS when drifting (TOO_WHITE / TOO_BLACK).
    - REVERSE is only a fallback recovery when completely lost off track.
    """
    # 1. On Edge -> Forward is top priority
    if prev_state == ON_EDGE:
        if action == FORWARD:
            return 15.0 if new_state == ON_EDGE else 2.0
        elif action == REVERSE:
            return -12.0 # Heavily penalize backing up when already on track
        else:
            return -3.0  # Mild penalty for turning when already centered

    # 2. Too far on white tape (Corner entry) -> MUST TURN (Never forward or reverse)
    elif prev_state == TOO_WHITE:
        if action in (LEFT, RIGHT) and new_state == ON_EDGE:
            return 15.0  # Turning to find edge at the corner gets top reward!
        elif action == FORWARD:
            return -10.0 # HEAVILY penalize driving forward through the tape at corners!
        elif action == REVERSE:
            return -8.0
        else:
            return -5.0

    # 3. Drifting into black background -> MUST TURN back to tape
    elif prev_state == TOO_BLACK:
        if action in (LEFT, RIGHT) and new_state in (ON_EDGE, TOO_WHITE):
            return 15.0  # Turning to recover tape/edge gets top reward!
        elif action == FORWARD:
            return -10.0 # HEAVILY penalize driving forward further into black!
        elif action == REVERSE:
            return -8.0
        else:
            return -6.0

    # 4. Completely lost off track (< 7 reflection)
    elif prev_state == LOST_OFFLINE:
        if new_state in (ON_EDGE, TOO_WHITE, TOO_BLACK):
            # Recovery achieved: Turning is preferred over reversing
            if action in (LEFT, RIGHT):
                return 16.0  # Turning around corner gets highest reward
            elif action == REVERSE:
                return 6.0   # Reversing recovery gets modest reward
            else:
                return 2.0
        else:
            # Still lost
            if action == REVERSE:
                return -2.0  # Negative (no free points for staying lost)
            elif action == FORWARD:
                return -15.0 # Heavy penalty for driving deeper into black
            else:
                return -6.0

    return -5.0

# =====================================================================
# 🚧 7. NON-RL OBSTACLE AVOIDANCE & PATH FINDING
# =====================================================================
def avoid_obstacle():
    """
    Non-RL bypass maneuver scaled for a wide (~20cm) EV3 robot:
    1. Stop and alert (beep)
    2. Full ~90° synchronized pivot into the open center arena
    3. Drive forward along the flank to completely clear the obstacle
    4. Full ~90° synchronized pivot back to face the track
    """
    print("🚧 Obstacle detected! Starting wide bypass maneuver...")
    ev3.speaker.beep(frequency=700, duration=150)
    left_motor.stop(Stop.BRAKE)
    right_motor.stop(Stop.BRAKE)
    wait(150)

    bypass_speed = 180
    pivot_angle  = 360   # Scaled for 20cm wheelbase: gives a true ~90 deg body turn
    flank_dist   = 600  # ~30cm travel to safely clear full robot body + obstacle
    aim_angle    = 340   # Re-aim turn angle back towards track

    # 1. Pivot Left ~90 deg into open center (for Counter-Clockwise inner edge)
    left_motor.run_angle(bypass_speed, -pivot_angle, wait=False)
    right_motor.run_angle(bypass_speed, pivot_angle, wait=True)
    wait(150)

    # 2. Drive forward past the obstacle
    left_motor.run_angle(bypass_speed, flank_dist, wait=False)
    right_motor.run_angle(bypass_speed, flank_dist, wait=True)
    wait(150)

    # 3. Pivot Right ~90 deg to face the line
    left_motor.run_angle(bypass_speed, aim_angle, wait=False)
    right_motor.run_angle(bypass_speed, -aim_angle, wait=True)
    wait(150)

def find_path():
    """
    Non-RL sweep routine to re-acquire the path after bypass or losing line:
    Drives forward/sweeps until the line's reflection threshold is detected.
    """
    print("🔍 Searching for line path...")
    # Drive forward towards the line (speed 140 deg/s ~6.8 cm/s)
    left_motor.run(140)
    right_motor.run(140)

    timeout = 0
    # Timeout 180 * 50ms = 9.0 seconds (~60cm max travel distance)
    while color_sensor.reflection() < 16 and timeout < 180:
        wait(50)
        timeout += 1

    # If not found directly, sweep in an arc to locate the line
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
print("🚀 Mode:", status_str)

while True:
    # 1. Non-RL Obstacle detection check
    if ir_sensor.distance() < OBSTACLE_THRESHOLD_CM:
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