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

    def randomize_lighting(self, ambient_range=(0.1, 0.5), diffuse_range=(0.3, 0.8), dir_noise=0.3):
        """
        Randomize lighting conditions for domain randomization.
        
        Simulates different outdoor lighting conditions:
        - Overcast (low diffuse, high ambient)
        - Direct sunlight (high diffuse, low ambient)
        - Different sun positions (direction noise)
        
        Args:
            ambient_range: (min, max) range for ambient light intensity
            diffuse_range: (min, max) range for diffuse light intensity
            dir_noise: Maximum deviation for light direction
        """
        nlight = self.env.Dcmm.model.nlight
        
        for i in range(nlight):
            # Randomize ambient light (uniform color)
            ambient = np.random.uniform(ambient_range[0], ambient_range[1])
            self.env.Dcmm.model.light_ambient[i] = [ambient, ambient, ambient]
            
            # Randomize diffuse light (can have slight color variation)
            diffuse_base = np.random.uniform(diffuse_range[0], diffuse_range[1])
            # Slight color temperature variation (warm vs cool light)
            color_temp = np.random.uniform(-0.1, 0.1)
            diffuse_r = diffuse_base + color_temp * 0.5  # Warm adds red
            diffuse_b = diffuse_base - color_temp * 0.5  # Cool adds blue
            diffuse_g = diffuse_base
            self.env.Dcmm.model.light_diffuse[i] = [
                np.clip(diffuse_r, 0, 1),
                np.clip(diffuse_g, 0, 1),
                np.clip(diffuse_b, 0, 1)
            ]
            
            # Randomize light direction (for directional lights)
            # Add noise to existing direction
            dir_original = self.env.Dcmm.model.light_dir[i].copy()
            dir_perturbation = np.random.uniform(-dir_noise, dir_noise, 3)
            new_dir = dir_original + dir_perturbation
            # Normalize direction vector
            new_dir = new_dir / (np.linalg.norm(new_dir) + 1e-6)
            self.env.Dcmm.model.light_dir[i] = new_dir

    def randomize_ground(self):
        """
        Randomize ground plane color for domain randomization.
        
        Simulates different ground conditions:
        - Dry soil (brown/tan)
        - Wet soil (darker brown)
        - Grass-covered (greenish)
        - Gravel/concrete (grey)
        """
        # Find floor geom - try common names
        floor_names = ["floor", "ground", "plane"]
        floor_id = -1
        
        for name in floor_names:
            floor_id = mujoco.mj_name2id(
                self.env.Dcmm.model, 
                mujoco.mjtObj.mjOBJ_GEOM, 
                name
            )
            if floor_id != -1:
                break
        
        if floor_id == -1:
            # Try finding by iterating geoms
            for i in range(self.env.Dcmm.model.ngeom):
                name = mujoco.mj_id2name(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                if name and ("floor" in name.lower() or "ground" in name.lower()):
                    floor_id = i
                    break
        
        if floor_id == -1:
            return  # No floor found
        
        # Randomly select ground type
        ground_type = np.random.choice(['soil_dry', 'soil_wet', 'grass', 'gravel'])
        
        if ground_type == 'soil_dry':
            # Brown/tan spectrum
            r = np.random.uniform(0.45, 0.60)
            g = np.random.uniform(0.35, 0.45)
            b = np.random.uniform(0.20, 0.30)
        elif ground_type == 'soil_wet':
            # Darker brown
            r = np.random.uniform(0.25, 0.40)
            g = np.random.uniform(0.20, 0.30)
            b = np.random.uniform(0.10, 0.20)
        elif ground_type == 'grass':
            # Greenish brown
            r = np.random.uniform(0.25, 0.40)
            g = np.random.uniform(0.40, 0.55)
            b = np.random.uniform(0.15, 0.25)
        else:  # gravel
            # Grey spectrum
            grey = np.random.uniform(0.4, 0.6)
            r = grey + np.random.uniform(-0.05, 0.05)
            g = grey + np.random.uniform(-0.05, 0.05)
            b = grey + np.random.uniform(-0.05, 0.05)
        
        self.env.Dcmm.model.geom_rgba[floor_id] = [r, g, b, 1.0]

    def apply_full_visual_dr(self):
        """
        Apply full visual domain randomization.
        
        Convenience method that applies all visual randomization at once:
        - Lighting randomization
        - Ground color randomization
        - Leaf/stem color randomization (already in randomize_plants)
        """
        self.randomize_lighting()
        self.randomize_ground()

    def randomize_stage2_avp_scene(self, use_extreme_distribution=False):
        """
        Stage 2 AVP dedicated initialization:
        - Plant + Fruit generation (same as Stage 1)
        - Set pre-grasp pose for maximum flexibility
        - Teleport robot based on EE-to-fruit distance
        - No occluders (Stage 1 handles obstacle avoidance)
        
        Args:
            use_extreme_distribution: 
                False = Phase 1 (90% reachable samples)
                True = Phase 2 (50% reachable + 50% extreme samples)
        """
        
        # ========================================
        # 1. Randomize Plants
        # ========================================
        self.randomize_plants()
        
        # ========================================
        # 2. Generate Fruit on Random Stem
        # ========================================
        stem_idx = np.random.randint(0, 5)
        stem_name = f"plant_stem_{stem_idx}"
        stem_body_id = mujoco.mj_name2id(
            self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, stem_name
        )
        
        if stem_body_id != -1:
            mocap_id = self.env.Dcmm.model.body_mocapid[stem_body_id]
            if mocap_id != -1:
                stem_pos = self.env.Dcmm.data.mocap_pos[mocap_id].copy()
            else:
                stem_pos = np.array([0.0, 1.0, 0.0])
        else:
            stem_pos = np.array([0.0, 1.0, 0.0])
        
        # Fruit height: 0.4m ~ 0.85m (within arm reachable range)
        fruit_height = np.random.uniform(0.4, 0.85)
        
        # Fruit position (slight offset from stem)
        offset_x = np.random.uniform(-0.05, 0.05)
        offset_y = np.random.uniform(-0.05, 0.05)
        fruit_pos = np.array([
            stem_pos[0] + offset_x, 
            stem_pos[1] + offset_y, 
            fruit_height
        ])
        
        self.env.object_pos3d = fruit_pos
        self.env.object_vel6d = np.zeros(6)
        self.env.object_q = np.array([1.0, 0.0, 0.0, 0.0])
        
        # ========================================
        # 3. Set Pre-grasp Pose and Compute EE Offset
        # ========================================
        pre_grasp_pose = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0])
        
        # Reset robot to origin to measure EE offset in robot frame
        self.env.Dcmm.data.qpos[0:3] = np.array([0, 0, 0])
        self.env.Dcmm.data.qpos[3:7] = np.array([1, 0, 0, 0])  # No rotation
        self.env.Dcmm.data.qpos[15:21] = pre_grasp_pose
        mujoco.mj_forward(self.env.Dcmm.model, self.env.Dcmm.data)
        
        # Root-to-EE offset in robot frame (yaw=0)
        root_pos = self.env.Dcmm.data.qpos[0:3].copy()
        ee_pos_local = self.env.Dcmm.data.body('link6').xpos.copy()
        ee_offset_local = ee_pos_local - root_pos  # [~0.05, ~0.30, ~0.41]
        
        # Forward reach in robot's +Y direction
        forward_reach = ee_offset_local[1]  # ~0.30m
        side_offset = ee_offset_local[0]    # ~0.05m (small)
        
        # ========================================
        # 4. Teleport Robot Based on EE Distance
        # ========================================
        # Compute target EE-to-fruit distance
        if use_extreme_distribution:
            # Phase 2: 50% reachable + 50% extreme
            if np.random.random() < 0.5:
                ee_to_fruit_dist = np.random.uniform(0.05, 0.25)  # Reachable
            else:
                ee_to_fruit_dist = np.random.uniform(0.25, 1.50)  # Extreme
        else:
            # Phase 1: 90% reachable + 10% edge
            if np.random.random() < 0.9:
                ee_to_fruit_dist = np.random.uniform(0.05, 0.25)  # Reachable
            else:
                ee_to_fruit_dist = np.random.uniform(0.25, 0.50)  # Edge
        
        # Approach angle: robot approaches from this direction towards fruit
        approach_angle = np.random.uniform(-np.pi, np.pi)
        
        # Robot faces the fruit: yaw = approach_angle + π
        yaw = approach_angle + np.pi
        
        # In world frame, when robot has yaw rotation, the EE offset transforms:
        # EE_world = Root_world + R(yaw) @ EE_local_xy + [0,0,Z]
        # Where R(yaw) is 2D rotation matrix
        
        # We want: ||EE_world - fruit_pos||_xy = ee_to_fruit_dist
        # (ignoring Z difference for placement calculation)
        
        # The EE will be at:
        #   EE_x = Root_x + cos(yaw)*side_offset - sin(yaw)*forward_reach
        #   EE_y = Root_y + sin(yaw)*side_offset + cos(yaw)*forward_reach
        
        # For EE to be at distance ee_to_fruit_dist from fruit, approaching from approach_angle:
        # Target EE position (on the line from fruit in approach direction):
        target_ee_x = fruit_pos[0] + ee_to_fruit_dist * np.cos(approach_angle)
        target_ee_y = fruit_pos[1] + ee_to_fruit_dist * np.sin(approach_angle)
        
        # Solve for root position:
        # Root_x = target_ee_x - cos(yaw)*side_offset + sin(yaw)*forward_reach
        # Root_y = target_ee_y - sin(yaw)*side_offset - cos(yaw)*forward_reach
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        robot_x = target_ee_x - cos_yaw * side_offset + sin_yaw * forward_reach
        robot_y = target_ee_y - sin_yaw * side_offset - cos_yaw * forward_reach
        robot_z = 0.0  # Base on ground
        
        # Set robot pose
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        robot_quat = np.array([cy, 0, 0, sy])
        
        self.env.Dcmm.data.qpos[0:3] = np.array([robot_x, robot_y, robot_z])
        self.env.Dcmm.data.qpos[3:7] = robot_quat
        self.env.Dcmm.data.qpos[15:21] = pre_grasp_pose
        
        # ========================================
        # 5. Visual Domain Randomization
        # ========================================
        self.apply_full_visual_dr()
        
        # Forward kinematics update
        mujoco.mj_forward(self.env.Dcmm.model, self.env.Dcmm.data)

