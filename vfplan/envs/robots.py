# Adapted from mocca_envs (UBC MOCCA lab). See THIRD_PARTY_NOTICES.md.
import os

import numpy as np

from vfplan.envs.bullet_utils import BodyPart, Joint, suppress_c_stdout

ASSETS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")


class WalkerBase:
    mirrored = False

    def apply_action(self, a):
        assert np.isfinite(a).all()
        x = np.clip(a, -1, 1)
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_torque(self.torque_limits[n] * float(x[n]))

    def calc_state(self, contact_object_ids=None):
        j = np.array(
            [j.current_relative_position() for j in self.ordered_joints],
            dtype=np.float32,
        )

        self.joint_angles = j[:, 0]
        self.joint_speeds = 0.1 * j[:, 1]  # Normalize
        self.joints_at_limit = np.count_nonzero(np.abs(self.joint_angles) > 0.99)

        body_pose = self.robot_body.pose()

        # Faster than computing the true CoM
        self.body_xyz = body_pose.xyz()

        if self.initial_z is None:
            self.initial_z = self.body_xyz[2]

        self.body_rpy = body_pose.rpy()
        roll, pitch, yaw = self.body_rpy

        rot = np.array(
            [
                [np.cos(-yaw), -np.sin(-yaw), 0],
                [np.sin(-yaw), np.cos(-yaw), 0],
                [0, 0, 1],
            ]
        )
        self.body_vel = np.dot(rot, self.robot_body.speed())
        vx, vy, vz = self.body_vel

        more = np.array(
            [self.body_xyz[2] - self.initial_z, vx, vy, vz, roll, pitch],
            dtype=np.float32,
        )

        if contact_object_ids is not None:
            for i, f in enumerate(self.feet):
                self.feet_xyz[i] = f.pose().xyz()
                contact_ids = set((x[2], x[4]) for x in f.contact_list())
                if contact_object_ids & contact_ids:
                    self.feet_contact[i] = 1.0
                else:
                    self.feet_contact[i] = 0.0

        state = np.concatenate((more, self.joint_angles, self.joint_speeds, self.feet_contact))

        return np.clip(state, -5, +5)

    def initialize(self):
        self.load_robot_model()
        self.make_robot_utils()

    def load_robot_model(self, model_path, flags, root_link_name=None):
        with suppress_c_stdout():
            self.object_id = self._p.loadMJCF(model_path, flags=flags)

        self.parse_joints_and_links(self.object_id)
        if root_link_name is not None:
            self.robot_body = self.parts[root_link_name]
        else:
            self.robot_body = BodyPart(self._p, "root", self.object_id, 0, -1)

        self.initialize_collision_matrix()

        self.feet = [self.parts[f] for f in self.foot_names]
        self.feet_contact = np.zeros(len(self.foot_names), dtype=np.float32)
        self.feet_xyz = np.zeros((len(self.foot_names), 3))
        self.calc_torque_limits()

    def calc_torque_limits(self):
        self.torque_limits = self.power * np.array(
            [self.power_coef[j.joint_name] for j in self.ordered_joints]
        )

    def make_robot_utils(self):
        # Convert between normalized joint angles (as observed) and radians.
        # weight is the range of motion, bias is the angle corresponding to -1.
        weight = np.array([j.upperLimit - j.lowerLimit for j in self.ordered_joints])
        bias = np.array([j.lowerLimit for j in self.ordered_joints])
        self.to_radians = lambda thetas: weight * (thetas + 1) / 2 + bias
        self.to_normalized = lambda angles: 2 * (angles - bias) / weight - 1

    def parse_joints_and_links(self, bodies):
        self.parts = {}
        self.jdict = {}
        self.ordered_joints = []
        bodies = [bodies] if np.isscalar(bodies) else bodies

        for i in range(len(bodies)):
            for j in range(self._p.getNumJoints(bodies[i])):
                self._p.setJointMotorControl2(
                    bodies[i],
                    j,
                    self._p.POSITION_CONTROL,
                    positionGain=0.1,
                    velocityGain=0.1,
                    force=0,
                )
                jointInfo = self._p.getJointInfo(bodies[i], j)
                joint_name = jointInfo[1].decode("utf8")
                part_name = jointInfo[12].decode("utf8")
                parent_index = jointInfo[16]

                self.parts[part_name] = BodyPart(self._p, part_name, bodies, i, j)
                self.parent_child_links[j] = parent_index

                if joint_name[:6] == "ignore":
                    Joint(self._p, joint_name, bodies, i, j).disable_motor()
                    continue

                if joint_name[:8] != "jointfix":
                    self.jdict[joint_name] = Joint(self._p, joint_name, bodies, i, j)
                    self.ordered_joints.append(self.jdict[joint_name])

    def initialize_collision_matrix(self):
        link_ids = [self.parts[part].bodyPartIndex for part in self.parts.keys()]

        # All pairs may collide by default...
        for i, link_a_id in enumerate(link_ids):
            for j in range(i, len(link_ids)):
                link_b_id = link_ids[j]
                self.collision_matrix[(link_a_id, link_b_id)] = True
                self.collision_matrix[(link_b_id, link_a_id)] = True

        # ...except a part and any of its ancestors.
        for part_id in link_ids:
            self.exclude_collision_with_all_ancestors(part_id)

        self.non_colliding_pairs = []
        for (part_a, part_b), can_collide in self.collision_matrix.items():
            if not can_collide and part_a != part_b:
                if (part_b, part_a) not in self.non_colliding_pairs:
                    self.non_colliding_pairs.append((part_a, part_b))

    def exclude_collision_with_all_ancestors(self, part_id):
        current_part = part_id
        while current_part in self.parent_child_links:
            parent_id = self.parent_child_links[current_part]
            self.collision_matrix[(part_id, parent_id)] = False
            self.collision_matrix[(parent_id, part_id)] = False
            current_part = parent_id

        if current_part == -1:
            for part in self.parts.values():
                self.collision_matrix[(part.bodyPartIndex, -1)] = False
                self.collision_matrix[(-1, part.bodyPartIndex)] = False

    def reset(self):
        self.feet_contact.fill(0.0)
        self.feet_xyz.fill(0.0)
        self.initial_z = None

        robot_state = self.calc_state()
        return robot_state


class Walker3D(WalkerBase):
    foot_names = ["right_foot", "left_foot"]
    power_amplifier = 1.2
    power_coef = {
        "abdomen_z": 60 * power_amplifier,
        "abdomen_y": 80 * power_amplifier,
        "abdomen_x": 60 * power_amplifier,
        "right_hip_x": 80 * power_amplifier,
        "right_hip_z": 60 * power_amplifier,
        "right_hip_y": 100 * power_amplifier,
        "right_knee": 90 * power_amplifier,
        "right_ankle": 60 * power_amplifier,
        "left_hip_x": 80 * power_amplifier,
        "left_hip_z": 60 * power_amplifier,
        "left_hip_y": 100 * power_amplifier,
        "left_knee": 90 * power_amplifier,
        "left_ankle": 60 * power_amplifier,
        "right_shoulder_x": 60 * power_amplifier,
        "right_shoulder_z": 60 * power_amplifier,
        "right_shoulder_y": 50 * power_amplifier,
        "right_elbow": 60 * power_amplifier,
        "left_shoulder_x": 60 * power_amplifier,
        "left_shoulder_z": 60 * power_amplifier,
        "left_shoulder_y": 50 * power_amplifier,
        "left_elbow": 60 * power_amplifier,
    }

    def __init__(self, bc):
        self._p = bc
        self.power = 1.0

        self.action_dim = 21
        # globals + angles + speeds + contacts
        self.state_dim = 6 + self.action_dim * 2 + 2

    def load_robot_model(self):
        flags = (
            self._p.MJCF_COLORS_FROM_FILE
            | self._p.URDF_USE_SELF_COLLISION
            | self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        )
        model_path = os.path.join(ASSETS_DIR, "walker3d.xml")

        self.collision_matrix = {}
        self.parent_child_links = {}

        super().load_robot_model(model_path, flags, root_link_name=None)

        # T-pose
        self.base_joint_angles = np.zeros(self.action_dim)
        self.base_position = (0, 0, 1.32)
        self.base_orientation = (0, 0, 0, 1)

    def set_base_pose(self, pose=None):
        self.base_joint_angles[:] = 0

        if pose == "running_start":
            self.base_joint_angles[[5, 6]] = -np.pi / 8  # Right leg
            self.base_joint_angles[10] = np.pi / 10  # Left leg back
            self.base_joint_angles[[13, 17]] = np.pi / 3  # Shoulder x
            self.base_joint_angles[[14]] = -np.pi / 6  # Right shoulder back
            self.base_joint_angles[[18]] = np.pi / 6  # Left shoulder forward
            self.base_joint_angles[[16, 20]] = np.pi / 3  # Elbow

    def reset(self, pos=None, quat=None, pose=None, vel=None):
        self.mirrored = False
        for i, j in enumerate(self.ordered_joints):
            j.reset_current_position(self.base_joint_angles[i], 0)

        if pose is not None:
            for i, j in enumerate(self.ordered_joints):
                j.reset_current_position(pose[i], 0)

        pos = self.base_position if pos is None else pos
        quat = self.base_orientation if quat is None else quat
        vel = np.zeros(3) if vel is None else vel

        self._p.resetBasePositionAndOrientation(self.object_id[0], posObj=pos, ornObj=quat)
        self._p.resetBaseVelocity(self.object_id[0], linearVelocity=vel)

        return super().reset()
