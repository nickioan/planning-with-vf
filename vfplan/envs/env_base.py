# Adapted from mocca_envs (UBC MOCCA lab). See THIRD_PARTY_NOTICES.md.
import numpy as np
import pybullet

from vfplan.envs.bullet_utils import BulletClient, Camera, Scene


class EnvBase:
    _render_width = 320 * 3
    _render_height = 240 * 3

    def __init__(self, robot_class, render=False):
        self.robot_class = robot_class
        self.scene = None
        self.physics_client_id = -1

        self.is_render = render
        self.initialize_scene_and_robot()

    @property
    def unwrapped(self):
        return self

    def close(self):
        if self.physics_client_id >= 0:
            self._p.disconnect()
        self.physics_client_id = -1

    def initialize_scene_and_robot(self):
        bc_mode = pybullet.GUI if self.is_render else pybullet.DIRECT
        self._p = BulletClient(connection_mode=bc_mode)

        if self.is_render:
            self.camera = Camera(self._p, 1 / self.control_step * self.llc_frame_skip)
            self.create_target()

        self.physics_client_id = self._p._client
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

        self.scene = Scene(
            self._p,
            gravity=9.81,
            timestep=self.control_step / self.llc_frame_skip / self.sim_frame_skip,
            frame_skip=self.sim_frame_skip,
        )

        self.robot = self.robot_class(self._p)
        self.robot.initialize()

        self.create_terrain()

    def render(self, distance=5.0, yaw=-60.0, pitch=-25.0):
        """Returns an RGB frame from a camera following the character (works headless)."""
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.robot.body_xyz,
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width) / self._render_height,
            nearVal=0.1,
            farVal=100.0,
        )
        renderer = pybullet.ER_BULLET_HARDWARE_OPENGL if self.is_render else pybullet.ER_TINY_RENDERER
        (_, _, px, _, _) = self._p.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=renderer,
        )
        rgb_array = np.reshape(np.array(px), (self._render_height, self._render_width, -1))
        return rgb_array[:, :, :3].astype(np.uint8)

    def reset(self):
        raise NotImplementedError

    def step(self, a):
        raise NotImplementedError

    def _handle_keyboard(self):
        keys = self._p.getKeyboardEvents()
        # Press "r" in the GUI to end the current episode.
        if ord("r") in keys and keys[ord("r")] == self._p.KEY_WAS_RELEASED:
            self.done = True
