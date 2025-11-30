"""
Constants and global variables for DcmmVecEnv.
Contains keyboard callback and global state variables.
"""

# Global control variables for keyboard input
paused = True
cmd_lin_y = 0.0
cmd_lin_x = 0.0
cmd_ang = 0.0
trigger_delta = False
trigger_delta_hand = False
delta_xyz = 0.0
delta_xyz_hand = 0.0

def env_key_callback(keycode):
    """Keyboard callback for manual control in viewer mode."""
    print("chr(keycode): ", (keycode))
    global cmd_lin_y, cmd_lin_x, cmd_ang, paused, trigger_delta, trigger_delta_hand, delta_xyz, delta_xyz_hand
    if keycode == 265: # AKA: up
        cmd_lin_y += 1
        print("up %f" % cmd_lin_y)
    if keycode == 264: # AKA: down
        cmd_lin_y -= 1
        print("down %f" % cmd_lin_y)
    if keycode == 263: # AKA: left
        cmd_lin_x -= 1
        print("left: %f" % cmd_lin_x)
    if keycode == 262: # AKA: right
        cmd_lin_x += 1
        print("right %f" % cmd_lin_x)
    if keycode == 52: # AKA: 4
        cmd_ang -= 0.2
        print("turn left %f" % cmd_ang)
    if keycode == 54: # AKA: 6
        cmd_ang += 0.2
        print("turn right %f" % cmd_ang)
    if chr(keycode) == ' ': # AKA: space
        if paused: paused = not paused
    if keycode == 334: # AKA + (on the numpad)
        trigger_delta = True
        delta_xyz = 0.1
    if keycode == 333: # AKA - (on the numpad)
        trigger_delta = True
        delta_xyz = -0.1
    if keycode == 327: # AKA 7 (on the numpad)
        trigger_delta_hand = True
        delta_xyz_hand = 0.2
    if keycode == 329: # AKA 9 (on the numpad)
        trigger_delta_hand = True
        delta_xyz_hand = -0.2

