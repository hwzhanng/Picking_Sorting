"""
Control management for DcmmVecEnv.
Handles action processing, control computation, and contact detection.
"""

import numpy as np
import mujoco
import copy
import configs.env.DcmmCfg as DcmmCfg


class ControlManager:
    """Manages control actions and contact detection for the environment."""

    def __init__(self, env):
        """
        Initialize control manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env

    def get_contacts(self):
        """
        Get contact information from the simulation.

        Returns:
            dict: Dictionary containing contact information for different components
        """
        plant_geom_ids = []
        leaf_geom_ids = []
        geom_ids = self.env.Dcmm.data.contact.geom
        geom1_ids = self.env.Dcmm.data.contact.geom1
        geom2_ids = self.env.Dcmm.data.contact.geom2

        ## get the contact points of the hand
        geom1_hand = np.where((geom1_ids < self.env.object_id) & (geom1_ids >= self.env.hand_start_id))[0]
        geom2_hand = np.where((geom2_ids < self.env.object_id) & (geom2_ids >= self.env.hand_start_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_hand.size != 0:
            contacts_geom1 = geom_ids[geom1_hand][:,1]
        if geom2_hand.size != 0:
            contacts_geom2 = geom_ids[geom2_hand][:,0]
        hand_contacts = np.concatenate((contacts_geom1, contacts_geom2))

        ## get the contact points of the object
        geom1_object = np.where((geom1_ids == self.env.object_id))[0]
        geom2_object = np.where((geom2_ids == self.env.object_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_object.size != 0:
            contacts_geom1 = geom_ids[geom1_object][:,1]
        if geom2_object.size != 0:
            contacts_geom2 = geom_ids[geom2_object][:,0]
        object_contacts = np.concatenate((contacts_geom1, contacts_geom2))

        ## get the contact points of the base
        geom1_base = np.where((geom1_ids == self.env.base_id))[0]
        geom2_base = np.where((geom2_ids == self.env.base_id))[0]
        contacts_geom1 = np.array([]); contacts_geom2 = np.array([])
        if geom1_base.size != 0:
            contacts_geom1 = geom_ids[geom1_base][:,1]
        if geom2_base.size != 0:
            contacts_geom2 = geom_ids[geom2_base][:,0]
        base_contacts = np.concatenate((contacts_geom1, contacts_geom2))

        ## get the contact points of the plant
        # Efficient approach: Iterate over all geoms once
        for i in range(self.env.Dcmm.model.ngeom):
            body_id = self.env.Dcmm.model.geom_bodyid[i]
            body_name = mujoco.mj_id2name(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name:
                if body_name.startswith("plant_stem"):
                    plant_geom_ids.append(i)
                elif body_name.startswith("plant_leaf"):
                    leaf_geom_ids.append(i)

        plant_contacts = np.array([])
        leaf_contacts = np.array([])

        for pid in plant_geom_ids:
            g1 = np.where((geom1_ids == pid))[0]
            g2 = np.where((geom2_ids == pid))[0]
            if g1.size != 0:
                plant_contacts = np.concatenate((plant_contacts, geom_ids[g1][:,1]))
            if g2.size != 0:
                plant_contacts = np.concatenate((plant_contacts, geom_ids[g2][:,0]))

        for lid in leaf_geom_ids:
            g1 = np.where((geom1_ids == lid))[0]
            g2 = np.where((geom2_ids == lid))[0]
            if g1.size != 0:
                leaf_contacts = np.concatenate((leaf_contacts, geom_ids[g1][:,1]))
            if g2.size != 0:
                leaf_contacts = np.concatenate((leaf_contacts, geom_ids[g2][:,0]))

        if self.env.print_contacts:
            print("object_contacts: ", object_contacts)
            print("hand_contacts: ", hand_contacts)
            print("base_contacts: ", base_contacts)
            print("plant_contacts: ", plant_contacts)
            print("leaf_contacts: ", leaf_contacts)

        return {
            "object_contacts": object_contacts,
            "hand_contacts": hand_contacts,
            "base_contacts": base_contacts,
            "plant_contacts": plant_contacts,
            "leaf_contacts": leaf_contacts
        }

    def update_target_ctrl(self):
        """Update target control values in the action buffer."""
        self.env.action_buffer["base"].append(copy.deepcopy(self.env.Dcmm.target_base_vel[:]))
        self.env.action_buffer["arm"].append(copy.deepcopy(self.env.Dcmm.target_arm_qpos[:]))
        # self.env.action_buffer["hand"].append(copy.deepcopy(self.env.Dcmm.target_hand_qpos[:]))

    def get_ctrl(self):
        """
        Map the action to the control commands.

        Returns:
            np.ndarray: Control commands for the robot
        """
        # Map the action to the control
        mv_steer, mv_drive = self.env.Dcmm.move_base_vel(self.env.action_buffer["base"][0]) # 8
        mv_arm = self.env.Dcmm.arm_pid.update(self.env.action_buffer["arm"][0],
                                             self.env.Dcmm.data.qpos[15:21],
                                             self.env.Dcmm.data.time) # 6
        # Keep hand fixed
        mv_hand = self.env.Dcmm.hand_pid.update(self.env.Dcmm.target_hand_qpos,
                                               self.env.Dcmm.data.qpos[21:37],
                                               self.env.Dcmm.data.time)
        ctrl = np.concatenate([mv_steer, mv_drive, mv_arm, mv_hand], axis=0)
        # Add Action Noise (Scale with self.k_act)
        ctrl *= np.random.normal(1, self.env.k_act, 30)
        if self.env.print_ctrl:
            print("##### ctrl:")
            print("mv_steer: {}, \nmv_drive: {}, \nmv_arm: {}, \nmv_hand: {}\n".format(
                mv_steer, mv_drive, mv_arm, mv_hand))
        return ctrl

    def step_mujoco_simulation(self, action_dict):
        """
        Execute one step of the MuJoCo simulation.

        Args:
            action_dict: Dictionary containing base, arm, and hand actions
        """
        # Construct raw action vector (18 dims)
        # Base (2)
        base_action = action_dict.get('base', np.zeros(2))
        # Arm (4)
        arm_action = action_dict.get('arm', np.zeros(4))
        if arm_action.size == 3:
            # Pad to 4 dims (x, y, z, yaw=0)
            arm_action = np.concatenate([arm_action, [0.0]])
        # Hand (12)
        if 'hand' in action_dict:
            hand_action = action_dict['hand']
            # If hand action is scalar or wrong shape, resize/pad
            if hand_action.size != 12:
                 hand_action = np.zeros(12)
        else:
            hand_action = np.zeros(12)

        raw_action = np.concatenate([base_action, arm_action, hand_action])

        # Apply LPF
        smoothed_action = self.env.alpha_lpf * raw_action + (1 - self.env.alpha_lpf) * self.env.prev_action
        self.env.prev_action = smoothed_action

        # Deconstruct smoothed action
        action_dict['base'] = smoothed_action[0:2]
        action_dict['arm'] = smoothed_action[2:6]
        action_dict['hand'] = smoothed_action[6:18]

        ## Update target base velocity
        self.env.Dcmm.target_base_vel[0:2] = action_dict['base']

        # Use only first 3 dims of arm action (x, y, z) for move_ee_pose
        action_arm = np.concatenate((action_dict["arm"][:3], np.zeros(3)))
        result_QP, _ = self.env.Dcmm.move_ee_pose(action_arm)
        if result_QP[1]:
            self.env.arm_limit = True
            self.env.Dcmm.target_arm_qpos[:] = result_QP[0]
        else:
            self.env.arm_limit = False

        # Force Hand Open
        self.env.Dcmm.target_hand_qpos[:] = self.env.Dcmm.open_hand_qpos[:]

        # Add Target Action to the Buffer
        self.update_target_ctrl()

        # Reset the Criteria for Successfully Touch
        self.env.step_touch = False

        for _ in range(self.env.steps_per_policy):
            # Lock hand to fixed open posture (Stage 1)
            self.env.Dcmm.data.qpos[21:33] = self.env.hand_open_angles

            # Update the control command according to the latest policy output
            self.env.Dcmm.data.ctrl[:] = self.get_ctrl()

            if self.env.render_per_step:
                # Rendering
                img = self.env.render()

            mujoco.mj_step(self.env.Dcmm.model, self.env.Dcmm.data)
            mujoco.mj_rnePostConstraint(self.env.Dcmm.model, self.env.Dcmm.data)

            # Update the contact information
            self.env.contacts = self.get_contacts()

            # Whether the base collides
            if self.env.contacts['base_contacts'].size != 0:
                self.env.terminated = True

            mask_coll = self.env.contacts['object_contacts'] < self.env.hand_start_id
            mask_finger = self.env.contacts['object_contacts'] > self.env.hand_start_id
            mask_hand = self.env.contacts['object_contacts'] >= self.env.hand_start_id
            mask_palm = self.env.contacts['object_contacts'] == self.env.hand_start_id

            # Whether the object is caught
            if self.env.step_touch == False:
                if self.env.task == "Catching" and np.any(mask_hand):
                    self.env.step_touch = True
                elif self.env.task == "Tracking" and np.any(mask_palm):
                    self.env.step_touch = True

            # Whether the object falls
            if not self.env.terminated:
                if self.env.task == "Catching":
                    self.env.terminated = np.any(mask_coll)
                elif self.env.task == "Tracking":
                    self.env.terminated = np.any(mask_coll) or np.any(mask_finger)

            # If the object falls, terminate the episode in advance
            if self.env.terminated:
                break

