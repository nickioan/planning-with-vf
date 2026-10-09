# Adapted from mocca_envs (UBC MOCCA lab), which builds on pybullet_envs.
# See THIRD_PARTY_NOTICES.md.
import contextlib
import ctypes
import functools
import inspect
import os
import sys
import time

import numpy as np
import pybullet


@contextlib.contextmanager
def suppress_c_stdout():
    """Silences output printed by pybullet's C code (e.g. while loading MJCF files)."""
    try:
        libc = ctypes.CDLL(None)
        sys.stdout.flush()
        libc.fflush(None)
        saved = os.dup(1)
    except (OSError, AttributeError, TypeError):
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        libc.fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class BulletClient:
    """A wrapper for pybullet that injects the physics client id into every call."""

    def __init__(self, connection_mode=pybullet.DIRECT):
        if connection_mode == pybullet.GUI:
            options = (
                "--background_color_red=1.0 --background_color_green=1.0 "
                "--background_color_blue=1.0 --width=1920 --height=1080"
            )
            self._client = pybullet.connect(connection_mode, options=options)
        else:
            self._client = pybullet.connect(connection_mode)

    def disconnect(self):
        if self._client >= 0:
            try:
                pybullet.disconnect(physicsClientId=self._client)
            except pybullet.error:
                pass
            # Client ids are reused; never disconnect twice.
            self._client = -1

    def __del__(self):
        self.disconnect()

    def __getattr__(self, name):
        attribute = getattr(pybullet, name)
        if inspect.isbuiltin(attribute) and name not in [
            "invertTransform",
            "multiplyTransforms",
            "getMatrixFromQuaternion",
            "getEulerFromQuaternion",
            "computeViewMatrixFromYawPitchRoll",
            "computeProjectionMatrixFOV",
            "getQuaternionFromEuler",
        ]:
            attribute = functools.partial(attribute, physicsClientId=self._client)
        return attribute


class PoseHelper:
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return pybullet.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()


class BodyPart:
    def __init__(self, bullet_client, body_name, bodies, bodyIndex, bodyPartIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        self.bp_pose = PoseHelper(self)

    def get_pose(self):
        body_id = self.bodies[self.bodyIndex]
        if self.bodyPartIndex == -1:
            (x, y, z), (a, b, c, d) = self._p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), *_ = self._p.getLinkState(body_id, self.bodyPartIndex)
        return np.array([x, y, z, a, b, c, d])

    def angular_speed(self):
        if self.bodyPartIndex == -1:
            _, (vr, vp, vy) = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            *_, (vr, vp, vy) = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1
            )
        return np.array([vr, vp, vy])

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = self._p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            *_, (vx, vy, vz), _ = self._p.getLinkState(
                self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1
            )
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return self._p.getContactPoints(bodyA=self.bodies[self.bodyIndex], linkIndexA=self.bodyPartIndex)


class Joint:
    def __init__(self, bullet_client, joint_name, bodies, bodyIndex, jointIndex):
        self.bodies = bodies
        self._p = bullet_client
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name

        jointInfo = self._p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.lowerLimit = jointInfo[8]
        self.upperLimit = jointInfo[9]

    def current_relative_position(self):
        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit)
        return (2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit), vel)

    def get_state(self):
        x, vx, _, _ = self._p.getJointState(self.bodies[self.bodyIndex], self.jointIndex)
        return x, vx

    def set_motor_torque(self, torque):
        self._p.setJointMotorControl2(
            bodyIndex=self.bodies[self.bodyIndex],
            jointIndex=self.jointIndex,
            controlMode=pybullet.TORQUE_CONTROL,
            force=torque,
        )

    def reset_current_position(self, position, velocity):
        self._p.resetJointState(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            targetValue=position,
            targetVelocity=velocity,
        )
        self.disable_motor()

    def disable_motor(self):
        self._p.setJointMotorControl2(
            self.bodies[self.bodyIndex],
            self.jointIndex,
            controlMode=pybullet.POSITION_CONTROL,
            targetPosition=0,
            targetVelocity=0,
            positionGain=0.1,
            velocityGain=0.1,
            force=0,
        )


class Scene:
    """Sets the physics parameters; the ground itself is a heightfield created by the env."""

    def __init__(self, bullet_client, gravity, timestep, frame_skip, num_solver_iterations=20):
        self._p = bullet_client
        self.timestep = timestep
        self.frame_skip = frame_skip
        self.dt = self.timestep * self.frame_skip

        self._p.setGravity(0, 0, -gravity)
        self._p.setDefaultContactERP(0.9)
        self._p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep * self.frame_skip,
            numSolverIterations=num_solver_iterations,
            numSubSteps=self.frame_skip,
        )

    def global_step(self):
        self._p.stepSimulation()


class Camera:
    def __init__(self, bc, fps=60):
        self._p = bc
        self._cam_dist = 4
        self._cam_yaw = -90
        self._cam_pitch = -75

        self._fps = fps
        self._target_period = 1 / (fps * 1.02)
        self._counter = time.perf_counter()

    def track(self, pos):
        # The camera is moved by the user; only throttle to real time.
        self.wait()

    def lookat(self, pos):
        self.camera_target = pos
        self._p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, pos)

    def wait(self):
        delta = time.perf_counter() - self._counter
        time.sleep(max(self._target_period - delta, 0))
        now = time.perf_counter()
        self._fps = 0.99 * self._fps + 0.01 / (now - self._counter)
        self._counter = now
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
