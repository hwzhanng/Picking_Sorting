"""
Randomization management for DcmmVecEnv.
Handles object, plant, physics, and environment randomization.
"""

import numpy as np
import mujoco
import xml.etree.ElementTree as ET
import configs.env.DcmmCfg as DcmmCfg


class RandomizationManager:
    """Manages all randomization aspects of the environment."""

    def __init__(self, env):
        """
        Initialize randomization manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env

    def reset_object(self):
        """
        Randomize object properties in the XML model.

        Returns:
            str: Modified XML string with randomized object properties
        """
        # Parse the XML string
        root = ET.fromstring(self.env.Dcmm.model_xml_string)

        # Find the <body> element with name="object"
        object_body = root.find(".//body[@name='object']")
        if object_body is not None:
            inertial = object_body.find("inertial")
            if inertial is not None:
                # Generate a random mass within the specified range
                self.env.random_mass = np.random.uniform(DcmmCfg.object_mass[0], DcmmCfg.object_mass[0])
                # Update the mass attribute
                inertial.set("mass", str(self.env.random_mass))
            joint = object_body.find("joint")
            if joint is not None:
                # Generate a random damping within the specified range
                random_damping = np.random.uniform(DcmmCfg.object_damping[0], DcmmCfg.object_damping[1])
                # Update the damping attribute
                joint.set("damping", str(random_damping))
            # Find the <geom> element
            geom = object_body.find(".//geom[@name='object']")
            if geom is not None:
                # Modify the type and size attributes
                object_id = np.random.choice([0, 1, 2, 3, 4])
                if self.env.object_train:
                    object_shape = DcmmCfg.object_shape[object_id]
                    geom.set("type", object_shape)  # Replace "box" with the desired type
                    object_size = np.array([np.random.uniform(low=low, high=high)
                                          for low, high in DcmmCfg.object_size[object_shape]])
                    geom.set("size", np.array_str(object_size)[1:-1])  # Replace with the desired size
                else:
                    object_mesh = DcmmCfg.object_mesh[object_id]
                    geom.set("mesh", object_mesh)
        # Convert the XML element tree to a string
        xml_str = ET.tostring(root, encoding='unicode')

        return xml_str

    def random_object_pose(self):
        """
        Randomize object pose.
        This method is now delegated to randomize_fruit_on_stem for plant scenario.
        Keeping for compatibility.
        """
        self.randomize_fruit_on_stem()

    def randomize_plants(self):
        """
        Randomize positions of 8 plant stems.
        CRITICAL: Enforce OCCLUSION. At least one plant must be between robot and fruit.
        """
        # Define spawn region: frontal cone
        min_plant_distance = 0.20  # Reduced slightly for higher density

        positions = []

        # Generate 8 random positions first
        for i in range(8):
            # Frontal cone: ±60° from +Y axis
            angle = np.random.uniform(-np.pi/3, np.pi/3)
            r = np.random.uniform(0.8, 2.0)
            x = r * np.sin(angle)
            y = r * np.cos(angle)
            positions.append([x, y])

        # Apply positions to mocap (temporarily)
        for i in range(8):
            stem_name = f"plant_stem_{i}"
            stem_body_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, stem_name)
            if stem_body_id != -1:
                mocap_id = self.env.Dcmm.model.body_mocapid[stem_body_id]
                if mocap_id != -1:
                    self.env.Dcmm.data.mocap_pos[mocap_id] = np.array([positions[i][0], positions[i][1], 0])

        # Domain Randomization: Physics (Stiffness/Damping) & Visuals (Color)
        # Randomize Leaf Physics
        stiffness_scale = np.random.uniform(0.2, 2.0)
        damping_scale = np.random.uniform(0.5, 1.5)

        for i in range(self.env.Dcmm.model.njnt):
            name = mujoco.mj_id2name(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and "leaf" in name:
                self.env.Dcmm.model.jnt_stiffness[i] = 0.5 * stiffness_scale
                # Damping is a DOF property, not Joint property directly in some bindings
                # Map Joint to DOF
                dof_adr = self.env.Dcmm.model.jnt_dofadr[i]
                self.env.Dcmm.model.dof_damping[dof_adr] = 0.05 * damping_scale

        # Randomize Leaf Colors (Visual DR)
        for i in range(self.env.Dcmm.model.ngeom):
            name = mujoco.mj_id2name(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and "leaf" in name:
                # Randomize green shade
                r_val = np.random.uniform(0.1, 0.4)
                g_val = np.random.uniform(0.3, 0.7)
                b_val = np.random.uniform(0.1, 0.3)
                self.env.Dcmm.model.geom_rgba[i] = np.array([r_val, g_val, b_val, 1.0])
            elif name and "stem" in name:
                # Randomize Stem Color (Brown/Green mix)
                is_brown = np.random.random() < 0.5
                if is_brown:
                    # Brownish: R>G, low B
                    r_val = np.random.uniform(0.3, 0.5)
                    g_val = np.random.uniform(0.2, 0.4)
                    b_val = np.random.uniform(0.0, 0.2)
                else:
                    # Greenish (like leaves but darker)
                    r_val = np.random.uniform(0.1, 0.3)
                    g_val = np.random.uniform(0.3, 0.5)
                    b_val = np.random.uniform(0.1, 0.2)
                self.env.Dcmm.model.geom_rgba[i] = np.array([r_val, g_val, b_val, 1.0])

    def randomize_fruit_and_occlusion(self):
        """
        1. Select a Target Stem.
        2. Select an Occluder Stem.
        3. Move Occluder Stem to block the path to Target Stem.
        4. Attach fruit to Target Stem.
        """
        # Select Target Stem (0-7)
        target_idx = np.random.randint(0, 8)
        target_stem_name = f"plant_stem_{target_idx}"

        # Select Occluder Stem (must be different)
        occluder_idx = (target_idx + 1) % 8
        occluder_stem_name = f"plant_stem_{occluder_idx}"

        # Get IDs
        target_body_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, target_stem_name)
        occluder_body_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, occluder_stem_name)

        target_mocap_id = self.env.Dcmm.model.body_mocapid[target_body_id]
        occluder_mocap_id = self.env.Dcmm.model.body_mocapid[occluder_body_id]

        # Get Target Position (already randomized in randomize_plants)
        target_pos = self.env.Dcmm.data.mocap_pos[target_mocap_id]

        # Calculate Robot Position (Base is at 0,0 roughly, or we use arm_base)
        robot_pos = self.env.Dcmm.data.body("arm_base").xpos

        # Calculate Occlusion Position
        # Place occluder at 50-80% of the distance to target
        ratio = np.random.uniform(0.5, 0.8)
        occluder_pos = robot_pos + ratio * (target_pos - robot_pos)
        occluder_pos[0] += np.random.uniform(-0.05, 0.05)
        occluder_pos[1] += np.random.uniform(-0.05, 0.05)
        occluder_pos[2] = 0 # Ground

        # Move Occluder
        self.env.Dcmm.data.mocap_pos[occluder_mocap_id] = occluder_pos

        # Now attach fruit to Target Stem
        stem_pos = target_pos
        height = np.random.uniform(0.8, 1.5)
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_y = np.random.uniform(-0.05, 0.05)

        fruit_x = stem_pos[0] + offset_x
        fruit_y = stem_pos[1] + offset_y

        self.env.object_pos3d = np.array([fruit_x, fruit_y, height])
        self.env.object_vel6d = np.zeros(6)
        self.env.object_q = np.array([1.0, 0.0, 0.0, 0.0])

    def randomize_fruit_on_stem(self):
        """
        Attach fruit (object) to a random position on a random plant stem.
        Since plants are now in frontal cone, fruit will also be in camera FOV.
        Maintains height constraints (0.8m - 1.5m) and distance from robot.
        """
        max_attempts = 20

        for attempt in range(max_attempts):
            # Select random stem (0-4)
            stem_idx = np.random.randint(0, 5)
            stem_name = f"plant_stem_{stem_idx}"

            # Get stem body and mocap IDs
            stem_body_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, stem_name)
            if stem_body_id != -1:
                mocap_id = self.env.Dcmm.model.body_mocapid[stem_body_id]
                if mocap_id == -1:
                    continue

                stem_pos = self.env.Dcmm.data.mocap_pos[mocap_id]

                # Random height on stem (0.8m - 1.5m above ground)
                height = np.random.uniform(0.8, 1.5)

                # Slight offset from stem center for realism
                offset_x = np.random.uniform(-0.05, 0.05)
                offset_y = np.random.uniform(-0.05, 0.05)

                fruit_x = stem_pos[0] + offset_x
                fruit_y = stem_pos[1] + offset_y

                # Validate reachability: distance from robot base
                fruit_distance = np.sqrt(fruit_x**2 + fruit_y**2)

                # Check if within reasonable reach (0.4m - 2.0m)
                if 0.4 <= fruit_distance <= 2.0 and 0.6 <= height <= 1.6:
                    self.env.object_pos3d = np.array([fruit_x, fruit_y, height])

                    # Static object, velocity is zero
                    self.env.object_vel6d = np.zeros(6)

                    # Fixed orientation (no rotation for sphere)
                    self.env.object_q = np.array([1.0, 0.0, 0.0, 0.0])
                    return  # Success

        # Fallback: if no valid position found, use last attempt
        self.env.object_pos3d = np.array([fruit_x, fruit_y, height])
        self.env.object_vel6d = np.zeros(6)
        self.env.object_q = np.array([1.0, 0.0, 0.0, 0.0])

    def random_PID(self):
        """Randomize PID controller parameters."""
        self.env.k_arm = np.random.uniform(0, 1, size=6)
        self.env.k_drive = np.random.uniform(0, 1, size=4)
        self.env.k_steer = np.random.uniform(0, 1, size=4)
        self.env.k_hand = np.random.uniform(0, 1, size=1)
        # Reset the PID Controller
        self.env.Dcmm.arm_pid.reset(self.env.k_arm*(DcmmCfg.k_arm[1]-DcmmCfg.k_arm[0])+DcmmCfg.k_arm[0])
        self.env.Dcmm.steer_pid.reset(self.env.k_steer*(DcmmCfg.k_steer[1]-DcmmCfg.k_steer[0])+DcmmCfg.k_steer[0])
        self.env.Dcmm.drive_pid.reset(self.env.k_drive*(DcmmCfg.k_drive[1]-DcmmCfg.k_drive[0])+DcmmCfg.k_drive[0])
        self.env.Dcmm.hand_pid.reset(self.env.k_hand[0]*(DcmmCfg.k_hand[1]-DcmmCfg.k_hand[0])+DcmmCfg.k_hand[0])

    def random_delay(self):
        """Randomize action delay buffer parameters."""
        self.env.action_buffer["base"].set_maxlen(np.random.choice(DcmmCfg.act_delay['base']))
        self.env.action_buffer["arm"].set_maxlen(np.random.choice(DcmmCfg.act_delay['arm']))
        self.env.action_buffer["hand"].set_maxlen(np.random.choice(DcmmCfg.act_delay['hand']))
        # Clear Buffer
        self.env.action_buffer["base"].clear()
        self.env.action_buffer["arm"].clear()
        self.env.action_buffer["hand"].clear()

    def randomize_stage2_scene(self):
        """
        Stage 2 Specific Randomization:
        1. Randomize Plants.
        2. Select Target Stem & Fruit.
        3. Teleport Robot Base (Domain Randomization).
        4. Place Occluder based on NEW robot position.
        5. Include "Bad" States (Singularity/Occlusion).
        """
        # 1. Randomize Plants (Standard)
        self.randomize_plants()

        # 2. Select Target Stem
        target_idx = np.random.randint(0, 8)
        target_stem_name = f"plant_stem_{target_idx}"
        target_body_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, target_stem_name)
        target_mocap_id = self.env.Dcmm.model.body_mocapid[target_body_id]
        target_pos = self.env.Dcmm.data.mocap_pos[target_mocap_id].copy()

        # 3. Teleport Robot Base
        # Determine "Front" direction relative to the stem
        # We want the robot to be roughly "facing" the stem/fruit, but the fruit is ON the stem.
        # Let's define the approach angle.
        
        # Decide if this is a "Bad" state (10% chance)
        is_bad_state = np.random.random() < 0.1
        
        if is_bad_state:
            # Bad State: Extreme distances or difficult angles
            case = np.random.randint(0, 3)
            if case == 0: # Too Far (Singularity risk)
                dist = np.random.uniform(1.0, 1.2)
            elif case == 1: # Too Close (Cramped)
                dist = np.random.uniform(0.35, 0.45)
            else: # Difficult Angle (Behind the stem?)
                dist = np.random.uniform(0.6, 0.8)
                # No specific angle logic yet, just rely on random angle potentially being bad?
                # Or maybe force it to be "behind" the fruit relative to some feature?
                pass
        else:
            # Normal State: Optimal working distance
            dist = np.random.uniform(0.6, 1.0)

        # Random Angle around the target
        # Since the stem is vertical, we pick an angle in the XY plane.
        # "Azimuth front 180 degrees" - let's assume this means we pick a random angle
        # and orient the robot to face the target.
        angle = np.random.uniform(-np.pi, np.pi)
        
        # Calculate Robot Base Position
        # Robot is placed at 'dist' away from target in direction 'angle'
        # So Robot Pos = Target Pos + Vector(dist, angle)
        # Wait, if we want the robot to *face* the target, the vector from Robot to Target is (cos(theta), sin(theta))
        # So Robot is at Target - Vector.
        
        robot_x = target_pos[0] + dist * np.cos(angle)
        robot_y = target_pos[1] + dist * np.sin(angle)
        
        # Robot Orientation (Yaw)
        # Robot should face the target.
        # Vector from Robot to Target: (target_x - robot_x, target_y - robot_y)
        # Yaw = atan2(dy, dx)
        yaw = np.arctan2(target_pos[1] - robot_y, target_pos[0] - robot_x)
        
        # Convert Yaw to Quaternion (w, x, y, z)
        # Axis: (0, 0, 1), Angle: yaw
        # q = [cos(yaw/2), 0, 0, sin(yaw/2)]
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        robot_quat = np.array([cy, 0, 0, sy])
        
        # Set Robot Base State (qpos 0-6)
        self.env.Dcmm.data.qpos[0:3] = np.array([robot_x, robot_y, 0.0]) # Z=0 for ground
        self.env.Dcmm.data.qpos[3:7] = robot_quat
        
        # 4. Place Occluder
        # Select Occluder Stem (must be different from target)
        occluder_idx = (target_idx + 1) % 8
        occluder_stem_name = f"plant_stem_{occluder_idx}"
        occluder_body_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, occluder_stem_name)
        occluder_mocap_id = self.env.Dcmm.model.body_mocapid[occluder_body_id]
        
        # Calculate Occluder Position
        # Place between Robot and Target
        # Ratio: 0.3 to 0.7 (closer to robot or closer to fruit)
        ratio = np.random.uniform(0.3, 0.7)
        
        # If "Bad" state (Occlusion), make it harder
        if is_bad_state and np.random.random() < 0.5:
             # Hard Occlusion: Directly in line, close to fruit
             ratio = np.random.uniform(0.7, 0.9)
        
        occluder_pos = np.array([robot_x, robot_y, 0]) + ratio * (target_pos - np.array([robot_x, robot_y, 0]))
        
        # Add some noise to occluder position so it's not always perfectly centered (unless bad state)
        if not is_bad_state:
            occluder_pos[0] += np.random.uniform(-0.1, 0.1)
            occluder_pos[1] += np.random.uniform(-0.1, 0.1)
            
        self.env.Dcmm.data.mocap_pos[occluder_mocap_id] = occluder_pos
        
        # 5. Attach Fruit to Target Stem
        # Random height on stem (0.8m - 1.5m)
        height = np.random.uniform(0.8, 1.5)
        
        # Orientation of fruit relative to stem
        # If we want the fruit to be "reachable", it should probably face the robot?
        # Or random? Real fruits are random.
        # Let's add a small offset from the stem center.
        offset_angle = np.random.uniform(-np.pi, np.pi)
        offset_r = 0.05 # 5cm radius from stem center
        
        fruit_x = target_pos[0] + offset_r * np.cos(offset_angle)
        fruit_y = target_pos[1] + offset_r * np.sin(offset_angle)
        
        self.env.object_pos3d = np.array([fruit_x, fruit_y, height])
        self.env.object_vel6d = np.zeros(6)
        self.env.object_q = np.array([1.0, 0.0, 0.0, 0.0])

