# Adapted from mocca_envs (UBC MOCCA lab). See THIRD_PARTY_NOTICES.md.
import os

import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets")
DEG2RAD = np.pi / 180


class VSphere:
    """Visual-only sphere (no collision), used to render targets and plans."""

    def __init__(self, bc, radius=None, pos=None, rgba=None):
        self._p = bc

        radius = 0.3 if radius is None else radius
        pos = (0, 0, 1) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else rgba

        shape = self._p.createVisualShape(
            self._p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            specularColor=[0.4, 0.4, 0],
        )
        self.id = self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=pos)
        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._rgba = rgba

    def set_position(self, pos=None):
        pos = self._pos if pos is None else pos
        self._p.resetBasePositionAndOrientation(self.id, posObj=pos, ornObj=self._quat)

    def set_color(self, rgba):
        t_rgba = tuple(rgba)
        if t_rgba != self._rgba:
            self._p.changeVisualShape(self.id, -1, rgbaColor=rgba)
            self._rgba = t_rgba


class AvoidanceZone:
    """Visual-only cylinder marking a circular no-entry zone (the "obstacle" task).

    It has no collision shape: entering it is detected by the planner/evaluation code.
    """

    def __init__(self, bc, radius=None, pos=None, rgba=None):
        self._p = bc

        radius = 0.3 if radius is None else radius
        pos = (0, 0, 1) if pos is None else pos
        rgba = (219 / 255, 72 / 255, 72 / 255, 1.0) if rgba is None else rgba
        self.rgba = rgba

        self.id = self._create(radius, pos)
        self._pos = pos
        self._quat = (0, 0, 0, 1)
        self._radius = radius

    def _create(self, radius, pos):
        shape = self._p.createVisualShape(
            self._p.GEOM_CYLINDER,
            radius=radius,
            length=2,
            rgbaColor=self.rgba,
            specularColor=[0, 0, 0],
        )
        return self._p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=pos)

    def set_position(self, pos=None):
        pos = self._pos if pos is None else pos
        self._p.resetBasePositionAndOrientation(self.id, posObj=pos, ornObj=self._quat)

    def update_shape(self, radius=None, pos=None):
        self._radius = self._radius if radius is None else radius
        self._pos = self._pos if pos is None else pos
        self._p.removeBody(self.id)
        self.id = self._create(self._radius, self._pos)


class Rectangle:
    """Static box with collision; used for hurdles and as the parts of a platform."""

    def __init__(self, bc, hdx, hdy, hdz, mass=0.0, pos=None, quat=None, rgba=None):
        self._p = bc

        self._dims = np.array([hdx, hdy, hdz], dtype=np.float32)
        self._pos = np.array([1.0, 1.0, 1.0]) if pos is None else pos
        self._quat = np.array([0.0, 0.0, 0.0, 1.0]) if quat is None else quat
        self.rgba = (55 / 255, 55 / 255, 55 / 255, 1) if rgba is None else rgba
        self.mass = mass

        self.id = self._create(self._dims)
        # Contact parameters the low-level controller was trained with.
        for link_id in range(-1, self._p.getNumJoints(self.id)):
            self._p.changeDynamics(
                self.id,
                link_id,
                lateralFriction=1.0,
                restitution=0.1,
                contactStiffness=30000,
                contactDamping=1000,
            )

    def _create(self, dims):
        box_shape = self._p.createCollisionShape(self._p.GEOM_BOX, halfExtents=dims)
        box_vshape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=dims,
            rgbaColor=self.rgba,
            specularColor=(0.4, 0.4, 0),
        )
        return self._p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=box_shape,
            baseVisualShapeIndex=box_vshape,
            basePosition=self._pos,
            baseOrientation=self._quat,
        )

    def set_position(self, pos=None, quat=None):
        self._pos = self._pos if pos is None else pos
        self._quat = self._quat if quat is None else quat
        self._p.resetBasePositionAndOrientation(self.id, posObj=self._pos, ornObj=self._quat)

    def update_shape(self, hdx=None, hdy=None, hdz=None):
        # Note: like the original code, the rebuilt body keeps pybullet's default
        # contact parameters (changeDynamics is only applied at construction).
        hdx = self._dims[0] if hdx is None else hdx
        hdy = self._dims[1] if hdy is None else hdy
        hdz = self._dims[2] if hdz is None else hdz
        self._dims = np.array([hdx, hdy, hdz], dtype=np.float32)
        self._p.removeBody(self.id)
        self.id = self._create(self._dims)


class Platform:
    """A step-up platform: a grey base with a thin black top plate."""

    def __init__(self, bc, hdx, hdy, hdz):
        self._p = bc

        black = (55 / 255, 55 / 255, 55 / 255, 1)
        grey = (110 / 255, 110 / 255, 110 / 255, 1)
        self.hdx = hdx
        self.hdy = hdy
        self.hdz = hdz
        self.dh = min(0.01, hdz / 2)
        self.hdz_base = hdz - self.dh

        self.basestep = Rectangle(self._p, hdx, hdy, self.hdz_base, pos=(0, 0, 0), rgba=grey)
        self.topstep = Rectangle(self._p, hdx, hdy, self.dh / 2, pos=(0, 0, 0), rgba=black)
        self._pos = (0, 0, 0)

    def set_position(self, pos=None, yaw=0):
        pos = self._pos if pos is None else pos
        self._pos = pos
        quat = self._p.getQuaternionFromEuler([0, 0, yaw])
        self.basestep.set_position(pos, quat)
        top_pos = (pos[0], pos[1], pos[2] + self.hdz_base + self.dh / 2)
        self.topstep.set_position(top_pos, quat)

    def update_shape(self, hdx=None, hdy=None, hdz=None):
        if hdx is not None:
            self.hdx = hdx
        if hdy is not None:
            self.hdy = hdy
        if hdz is not None:
            self.hdz = hdz
            self.dh = min(0.01, hdz / 2)
            self.hdz_base = hdz - self.dh
        self.basestep.update_shape(self.hdx, self.hdy, self.hdz_base)
        self.topstep.update_shape(self.hdx, self.hdy, self.dh / 2)


class FlatGround:
    """Flat heightfield ground plus the procedural footstep trajectory generator."""

    def __init__(self, bc, height_field_size, n_steps=50):
        self._p = bc
        self.height_field_size = height_field_size
        self.id = -1
        self.shape_id = -1
        self.n_steps = n_steps

        texture_file = os.path.join(ASSETS_DIR, "checker_blue.png")
        self.texture_id = self._p.loadTexture(texture_file)
        self.texture_scaling = 10

        # Alternating left/right lateral offsets of consecutive footsteps.
        self.base_phi = DEG2RAD * np.array([-10] + [20, -20] * (self.n_steps // 2 - 1) + [10])
        self.total_length = 60

    def reset_trajectory(self, yaw_ratio=0.0, height_scale=0.0):
        """Creates the ground (on first call) and returns a new footstep trajectory.

        Returns an array of shape (n_steps, 6): x, y, z, yaw, x_tilt, y_tilt.
        """
        hfield = np.ones(self.height_field_size)
        rows, cols = self.height_field_size
        self.hfield = hfield
        self.height_scale = height_scale

        hfield_data = hfield.reshape(rows * cols) * height_scale
        offset = -(hfield_data.max() + hfield_data.min()) / 2

        if self.id == -1:
            self.shape_id = self._p.createCollisionShape(
                shapeType=self._p.GEOM_HEIGHTFIELD,
                meshScale=[self.total_length / (rows // 2), self.total_length / (cols // 2), 1],
                heightfieldTextureScaling=self.texture_scaling,
                heightfieldData=hfield_data,
                numHeightfieldRows=rows,
                numHeightfieldColumns=cols,
            )
            self.id = self._p.createMultiBody(0, self.shape_id, -1, (0, 0, -offset))
            self._p.changeDynamics(
                self.id,
                -1,
                lateralFriction=1.0,
                restitution=0.2,
                contactStiffness=30000,
                contactDamping=1000,
            )

        self._p.changeVisualShape(
            self.id,
            -1,
            textureUniqueId=self.texture_id,
            rgbaColor=[1, 1, 1, 1],
            specularColor=[0, 0, 0],
        )
        return self.generate_step_placements(self.hfield, self.height_scale, yaw_ratio)

    def generate_step_placements(self, hfield, height_scale, yaw_ratio=0.0):
        """Procedural footstep generator (paper Supp. B.1).

        `yaw_ratio` in [0, 1] scales the maximum per-step heading change (30 deg * yaw_ratio).
        """
        rows, cols = self.height_field_size
        n_steps = self.n_steps

        # The random draws below only fill the first three entries, which are then
        # overwritten; they are kept because they advance the global numpy RNG.
        y_range = np.array([-10, 10]) * DEG2RAD
        p_range = np.array([-30, 30]) * DEG2RAD + np.pi / 2
        t_range = np.array([-0, 0]) * DEG2RAD

        dr = np.random.uniform(0.65, 1.25, size=n_steps)
        dphi = np.random.uniform(*y_range, size=n_steps)
        dtheta = np.random.uniform(*p_range, size=n_steps)
        # First step below the feet, the next two slightly further ahead.
        dr[0], dphi[0], dtheta[0] = 0.0, 0, np.pi / 2
        dr[1], dphi[1], dtheta[1] = 0.75, 0, np.pi / 2
        dr[2], dphi[2], dtheta[2] = 0.75, 0.0, np.pi / 2

        x_tilt = np.random.uniform(*t_range, size=n_steps)
        y_tilt = np.random.uniform(*t_range, size=n_steps)
        x_tilt[0:3] = 0
        y_tilt[0:3] = 0

        dphi = np.cumsum(dphi)
        x_ = dr * np.sin(dtheta) * np.cos(dphi)
        y_ = dr * np.sin(dtheta) * np.sin(dphi)
        z_ = dr * np.cos(dtheta)

        x = np.cumsum(x_)
        y = np.cumsum(y_)
        z = np.cumsum(z_)

        terrain_info = np.stack((x, y, z, dphi, x_tilt, y_tilt), axis=1)

        length_per_index = self.total_length / (rows // 2)
        for i in range(0, 3):
            x_index = int(terrain_info[i, 0] / length_per_index) + (rows // 2)
            y_index = int(terrain_info[i, 1] / length_per_index) + (cols // 2)

            terrain_info[i, 2] = hfield[y_index, x_index] * height_scale
            terrain_info[i, 4] = np.arctan2(
                (hfield[y_index, x_index + 1] - hfield[y_index, x_index - 1]) * height_scale,
                length_per_index * 2,
            )
            terrain_info[i, 5] = np.arctan2(
                (hfield[y_index + 1, x_index] - hfield[y_index - 1, x_index]) * height_scale,
                length_per_index * 2,
            )

        dr_min = 0.5
        dr_max = 1.125
        max_yaw = 30 * yaw_ratio
        for i in range(0, n_steps - 1):
            yaw_scale = np.random.uniform(-max_yaw, max_yaw)
            next_step_xyz = terrain_info[i]
            bound_checked_index = (i + 1) % n_steps

            base_yaw = terrain_info[i, 3]
            base_phi = self.base_phi[bound_checked_index]

            dr = np.random.uniform(dr_min, dr_max, 1)[0]
            if i < 2:
                dr = 0.75
                yaw_scale = 0.0

            yaw = yaw_scale * DEG2RAD
            dx = dr * np.cos(yaw + base_phi)
            dy = dr * np.sin(yaw - base_phi)

            matrix = np.array(
                [
                    [np.cos(base_yaw), -np.sin(base_yaw)],
                    [np.sin(base_yaw), np.cos(base_yaw)],
                ]
            )
            dxy = np.dot(matrix, np.concatenate(([dx], [dy])))

            x = next_step_xyz[0] + dxy[0]
            y = next_step_xyz[1] + dxy[1]
            x_index = int(x / length_per_index) + (rows // 2)
            y_index = int(y / length_per_index) + (cols // 2)
            z = hfield[y_index, x_index] * height_scale
            y_tilt = np.arctan2(
                (hfield[y_index + 1, x_index] - hfield[y_index - 1, x_index]) * height_scale,
                length_per_index * 2,
            )
            x_tilt = np.arctan2(
                (hfield[y_index, x_index + 1] - hfield[y_index, x_index - 1]) * height_scale,
                length_per_index * 2,
            )
            matrix = np.array(
                [
                    [-np.cos(base_yaw + yaw), -np.sin(base_yaw + yaw)],
                    [np.sin(base_yaw + yaw), -np.cos(base_yaw + yaw)],
                ]
            )
            x_tilt, y_tilt = np.dot(matrix, np.concatenate(([-y_tilt], [x_tilt])))

            terrain_info[bound_checked_index, 0] = x
            terrain_info[bound_checked_index, 1] = y
            terrain_info[bound_checked_index, 2] = z
            terrain_info[bound_checked_index, 3] = yaw + base_yaw
            terrain_info[bound_checked_index, 4] = x_tilt
            terrain_info[bound_checked_index, 5] = y_tilt
        return terrain_info
