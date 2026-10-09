# Adapted from mocca_envs (UBC MOCCA lab). See THIRD_PARTY_NOTICES.md.
"""Humanoid footstep-following environment used for all tasks in the paper.

The low-level ALLSTEPS controller is driven through the last 10 entries of the
observation (the next two target footsteps, relative to the root). Planners
overwrite those entries with their own plan; see `vfplan.rollout`.
"""

import numpy as np
from scipy.interpolate import splev, splprep

from vfplan.envs.bullet_objects import (
    AvoidanceZone,
    FlatGround,
    Platform,
    Rectangle,
    VSphere,
)
from vfplan.envs.env_base import EnvBase
from vfplan.envs.robots import Walker3D

TASKS = ("flat", "platform", "hurdle", "obstacle", "composed", "sequential")

DODGERBLUE = (0.11764705882352941, 0.5647058823529412, 1.0, 1.0)


class Walker3DPlanningEnv(EnvBase):
    """Walker3D following a procedurally generated footstep trajectory.

    Args:
        task: which objects to place along the trajectory.
            "flat"      - nothing; waypoints are sampled along the trajectory.
            "platform"  - step-up platforms (paper: platform task).
            "hurdle"    - thin walls to step over (paper: hurdle task).
            "obstacle"  - circular no-entry zones (paper: obstacle-avoidance task).
            "composed"  - platforms with a hurdle on top (paper Sec. 7.4).
            "sequential"- fixed course mixing all three (demo video).
        platform_height_range: platform heights are drawn from 12 evenly spaced
            values in this range [m].
        hurdle_heights: hurdle heights are drawn from these values [m].
        obstacle_radius_range: avoidance-zone radii are drawn uniformly from this range [m].
    """

    control_step = 1 / 60
    llc_frame_skip = 1
    sim_frame_skip = 4

    def __init__(
        self,
        task="platform",
        render=False,
        platform_height_range=(0.10, 0.65),
        hurdle_heights=(0.25, 0.30, 0.35),
        obstacle_radius_range=(1.5, 1.5),
    ):
        assert task in TASKS, f"Unknown task {task!r}, expected one of {TASKS}"
        self.task = task
        self.platform_height_range = tuple(platform_height_range)
        self.hurdle_heights = tuple(hurdle_heights)
        self.obstacle_radius_range = tuple(obstacle_radius_range)

        self.done_height = 0.7
        self.yaw_avg = 0.0
        self.tau = 0.98
        self.mean_targets = None
        self.viz_k = None
        self.rendered_step_count = 4
        self.min_grid = (9, 35)
        self.time_index = 0
        self.n_steps = 60
        self.max_curriculum = 5
        self.curriculum = 1

        self.n_hurdles = 12
        self.n_flat_waypoints = 7
        self.n_obstacles = 5
        self.n_platforms = 5

        self.got_new_step = False

        super().__init__(Walker3D, render)
        self.robot.set_base_pose(pose="running_start")

        self.electricity_cost = 4.5
        self.stall_torque_cost = 0.225
        self.joints_at_limit_cost = 0.1

        self.lookahead = 2
        self.next_step_index = 0

        # Egocentric heightmap window [m]: 0.5 m behind to 4.5 m ahead, 3 m to each side.
        self.hm_x_range = (-0.5, 4.5)
        self.hm_y_range = (-3.0, 3.0)
        self.hm_resolution = (32, 32)
        # World-frame bounding box of the last heightmap (used to filter collected data).
        self.hm_x_window = np.array(self.hm_x_range)
        self.hm_y_window = np.array(self.hm_y_range)

        self.foot_dist_to_target = np.zeros(len(self.robot.feet), dtype=np.float32)

        # x, y, z, phi, x_tilt, y_tilt
        self.terrain_info = np.zeros((self.min_grid[1], 6))

        # robot state + (2 targets) * (x, y, z, x_tilt, y_tilt)
        self.observation_dim = self.robot.state_dim + self.lookahead * 5
        self.action_dim = self.robot.action_dim

    # ------------------------------------------------------------------ terrain

    def create_terrain(self):
        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        self.platforms = []
        self.hurdles = []
        self.obstacles = []

        task = self.task
        if task == "obstacle":
            for _ in range(self.n_obstacles):
                self.obstacles.append(AvoidanceZone(self._p, rgba=(1.0, 0.0, 0.0, 1.0)))

        if task == "platform":
            for _ in range(self.n_platforms):
                lo, hi = self.platform_height_range
                self.platforms.append(Platform(self._p, 1.0, 1.0, np.random.uniform(lo, hi)))

        if task == "hurdle":
            for _ in range(self.n_hurdles):
                self.hurdles.append(Rectangle(self._p, 0.001, 1.0, 0.30, rgba=(1, 0, 0, 1)))

        if task == "composed":
            for _ in range(self.n_platforms):
                lo, hi = self.platform_height_range
                self.platforms.append(Platform(self._p, 1.0, 1.0, np.random.uniform(lo, hi)))
                self.hurdles.append(Rectangle(self._p, 0.001, 0.9, 0.20, rgba=(1, 0, 0, 1)))

        if task == "sequential":
            for height in [0.6, 0.55]:
                self.platforms.append(Platform(self._p, 1.0, 1.0, height))
            for _ in range(3):
                self.hurdles.append(Rectangle(self._p, 0.001, 0.9, 0.30, rgba=(1, 0, 0, 1)))
            for _ in range(2):
                self.obstacles.append(AvoidanceZone(self._p, radius=1.5))

        self.ground = FlatGround(self._p, (256, 256), n_steps=self.n_steps)

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)

    def _trajectory_spline(self):
        xy = [self.terrain_info[:, 0], self.terrain_info[:, 1]]
        tck, _ = splprep(xy, s=1)
        return tck

    def _final_waypoint(self):
        return np.array([self.terrain_info[-1, 0], self.terrain_info[-1, 1], 1.0])

    def randomize_terrain(self, randomize=True):
        cur_ratio = self.curriculum / self.max_curriculum
        task = self.task

        if randomize:
            self.terrain_info = self.ground.reset_trajectory(cur_ratio, height_scale=1.0)

        if task == "platform" and randomize:
            # Platforms are placed along a B-spline fit of the footstep trajectory.
            self.platform_locations = []
            tck = self._trajectory_spline()
            u_new = np.linspace(0.125, 0.775, self.n_platforms)
            u_new[1:] += np.random.uniform(-0.05, 0.05, self.n_platforms - 1)
            x_new, y_new = splev(u_new, tck)

            heights = np.linspace(*self.platform_height_range, 12)
            for i in range(self.n_platforms):
                pos = (x_new[i], y_new[i], 1.0)
                np.random.uniform(-45, 45)  # unused draw, kept to preserve the RNG stream
                self.platforms[i].update_shape(hdz=np.random.choice(heights))
                self.platforms[i].set_position(pos)
                wpt = np.copy(pos)
                wpt[2] += self.platforms[i].hdz
                self.platform_locations.append(wpt)

        if task == "hurdle":
            # Hurdles are placed perpendicular to the trajectory.
            self.hurdle_locations = []
            self.hurdle_angles = []
            self.waypoint_locations = []
            tck = self._trajectory_spline()
            u_new = np.linspace(0.125, 0.75, self.n_hurdles)
            u_new[1:] += np.random.uniform(0, 0.05, self.n_hurdles - 1)
            x_new, y_new = splev(u_new, tck)
            dh = 1e-4
            self.hurdle_heights_current = np.random.choice(self.hurdle_heights, self.n_hurdles)
            for i in range(self.n_hurdles):
                x_a, y_a = splev(u_new[i] + dh, tck)
                theta = np.arctan2(y_a - y_new[i], x_a - x_new[i])
                quat = self._p.getQuaternionFromEuler([0, 0, theta])
                pos = (x_new[i], y_new[i], 1.0)
                self.hurdles[i].update_shape(hdz=self.hurdle_heights_current[i])
                self.hurdles[i].set_position(pos, quat=quat)
                self.hurdle_locations.append(np.copy(pos))
                self.hurdle_angles.append(theta)

                if i < self.n_hurdles - 1:
                    x_wpt, y_wpt = splev((u_new[i] + u_new[i + 1]) / 2, tck)
                    self.waypoint_locations.append(np.array([x_wpt + 0.1, y_wpt, 1.0]))
                else:
                    self.waypoint_locations.append(self._final_waypoint())

        if task == "flat":
            self.waypoint_locations = []
            tck = self._trajectory_spline()
            u_new = np.linspace(0.125, 0.775, self.n_hurdles)
            u_new += np.random.uniform(-0.1, 0.1, self.n_hurdles)
            u_new = np.sort(u_new)
            x_new, y_new = splev(u_new, tck)
            for i in range(self.n_flat_waypoints):
                self.waypoint_locations.append(np.array([x_new[i], y_new[i], 1.0]))
            self.waypoint_locations.append(self._final_waypoint())

        if task == "obstacle" and randomize:
            # Waypoints sit halfway between consecutive avoidance zones.
            self.waypoint_locations = []
            self.obstacle_locations = []
            self.obstacle_radius = []
            tck = self._trajectory_spline()
            u_new = np.linspace(0.125, 0.65, self.n_obstacles)
            x_new, y_new = splev(u_new, tck)
            r_lo, r_hi = self.obstacle_radius_range
            for i in range(self.n_obstacles):
                pos = np.array([x_new[i], y_new[i], 1.0])
                radius = r_lo if r_lo == r_hi else np.random.uniform(r_lo, r_hi)
                self.obstacles[i].update_shape(radius, pos)
                self.obstacle_locations.append(pos)
                self.obstacle_radius.append(radius)
            for i in range(self.n_obstacles - 1):
                pos1 = self.obstacle_locations[i]
                pos2 = self.obstacle_locations[i + 1]
                self.waypoint_locations.append((pos1 + pos2) / 2)
            self.waypoint_locations.append(self._final_waypoint())

        if task == "composed":
            self.waypoints = []
            self.hurdle_locations = []
            self.hurdle_angles = []
            tck = self._trajectory_spline()
            u_new = np.linspace(0.125, 0.775, self.n_platforms)
            u_new[1:] += np.random.uniform(-0.05, 0.05, self.n_platforms - 1)
            x_new, y_new = splev(u_new, tck)

            for i in range(self.n_platforms):
                pos = (x_new[i], y_new[i], 1.0)
                self.platforms[i].update_shape(hdz=0.4)
                self.platforms[i].set_position(pos)
                wpt = np.copy(pos)
                wpt[2] += self.platforms[i].hdz
                self.waypoints.append(np.copy(wpt))
                theta = np.random.uniform(-np.pi / 8, np.pi / 8)
                quat = self._p.getQuaternionFromEuler([0, 0, theta])
                self.hurdles[i].set_position(wpt, quat)
                self.hurdle_locations.append(np.copy(wpt))
                self.hurdle_angles.append(theta)
            self.waypoints.append([self.terrain_info[-1, 0], self.terrain_info[-1, 1], 1.0])

        if task == "sequential":
            self.hurdle_locations = []
            self.hurdle_angles = []
            self.obstacle_locations = []
            self.obstacle_radius = []
            tck = self._trajectory_spline()
            u_platform = [0.125, 0.7]
            x_platform, y_platform = splev(u_platform, tck)
            u_hurdle = [0.35, 0.425, 0.6]
            x_hurdle, y_hurdle = splev(u_hurdle, tck)
            u_obstacle = [0.25, 0.525]
            x_obstacle, y_obstacle = splev(u_obstacle, tck)

            for i in range(len(u_platform)):
                self.platforms[i].set_position((x_platform[i], y_platform[i], 1.0))
            thetas = [-np.pi / 8, np.pi / 8, np.pi / 8]
            for i in range(len(u_hurdle)):
                pos = (x_hurdle[i], y_hurdle[i], 1.0)
                quat = self._p.getQuaternionFromEuler([0, 0, thetas[i]])
                self.hurdles[i].set_position(pos, quat)
                self.hurdle_locations.append(np.copy(pos))
                self.hurdle_angles.append(thetas[i])
            for i in range(len(u_obstacle)):
                pos = (x_obstacle[i], y_obstacle[i], 1.0)
                self.obstacles[i].set_position(pos)
                self.obstacle_locations.append(np.copy(pos))
                self.obstacle_radius.append(1.5)

            self.waypoints = np.array(
                [
                    [x_platform[0], y_platform[0], 1.6],
                    [x_hurdle[0], y_hurdle[0], 1.0],
                    [x_hurdle[1], y_hurdle[1], 1.0],
                    [x_hurdle[2], y_hurdle[2], 1.0],
                    [x_platform[1], y_platform[1], 1.55],
                    [self.terrain_info[-1, 0], self.terrain_info[-1, 1], 1.0],
                ]
            )

        self.project_trajectory()

    def project_trajectory(self):
        """Sets the height of every trajectory footstep to the height of whatever is below it."""
        self.disable_humanoid_collisions()
        heights, _ = self.get_height_from_coordinate(self.terrain_info)
        self.enable_humanoid_collisions()
        self.terrain_info[:, 2] = np.copy(heights)

    def set_platform_heights(self, height):
        """Overrides the height of every platform (used for per-height evaluation)."""
        for i, platform in enumerate(self.platforms):
            platform.update_shape(hdz=height)
            platform.set_position()
            self.platform_locations[i][2] = 1.0 + height
        self.project_trajectory()

    def update_curriculum(self, val):
        self.curriculum = min(val, self.max_curriculum)

    # ---------------------------------------------------------------- reset/step

    def reset(self, randomize=True):
        self.yaw_avg = 0.0
        self.foot_placement = []
        self.stepping_time_index = []
        self.energy_cost = [0.0]
        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

        self.robot.set_base_pose(pose="running_start")
        self.done = False
        self.target_reached_count = 0
        self.time_index = 0
        self.swing_leg = 1 if self.robot.mirrored else 0

        self.randomize_terrain(randomize)
        height = self.terrain_info[0][2] + 1.32
        self.robot_state = self.robot.reset(pos=(0.3, 0, height), vel=[0.0, 0, 0])
        self.calc_feet_state()

        self.next_step_index = 1

        if self.is_render:
            self.camera.lookat(self.robot.body_xyz + [3, 0, 0])

        self.targets = self.delta_to_k_targets(k=self.lookahead)
        self.calc_potential()

        state = np.concatenate((self.robot_state, self.targets.flatten()))
        state[0] = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])

        if self.is_render:
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        return state

    def step(self, action):
        self.time_index += 1
        self.robot.apply_action(action)
        self.scene.global_step()

        self.robot_state = self.robot.calc_state()
        self.calc_env_state(action)
        self.energy_cost.append(self.energy_penalty)

        _, _, yaw = self.robot.body_rpy
        self.yaw_avg = self.tau * self.yaw_avg + (1 - self.tau) * yaw

        reward = 1.0 * self.progress - self.energy_penalty
        reward += self.step_bonus + self.target_bonus - self.speed_penalty
        reward += self.tall_bonus - self.posture_penalty - self.joints_penalty

        state = np.concatenate((self.robot_state, self.targets.flatten()))

        if self.is_render:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)
            self._draw_plan()

        state[0] = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        return state, reward, self.done, {}

    # ------------------------------------------------------------ observations

    def get_scandot(self):
        """Egocentric heightmap aligned with the direction to the current waypoint.

        Returns (heightmap, mask, angle_to_waypoint); heightmap is in world z.
        """
        nx, ny = self.hm_resolution
        x_offset = self.robot.body_xyz[0]
        y_offset = self.robot.body_xyz[1]

        deltas = self.waypoint[0:3] - self.robot.body_xyz
        angle_to_wpt = np.arctan2(deltas[1], deltas[0])
        rotation = np.array(
            [
                [np.cos(angle_to_wpt), np.sin(angle_to_wpt)],
                [-np.sin(angle_to_wpt), np.cos(angle_to_wpt)],
            ]
        )

        x = np.linspace(self.hm_x_range[0], self.hm_x_range[1], nx)
        y = np.linspace(self.hm_y_range[0], self.hm_y_range[1], ny)
        xv, yv = np.meshgrid(x, y, indexing="ij")
        xv, yv = np.einsum("ij, mni -> jmn", rotation, np.dstack([xv, yv]))
        xv = xv + x_offset
        yv = yv + y_offset
        self.hm_x_window = np.array([np.min(xv), np.max(xv)])
        self.hm_y_window = np.array([np.min(yv), np.max(yv)])

        heightmap = np.empty_like(xv)
        mask = np.empty_like(xv)

        # Hide the humanoid and the hurdles from the rays: the diffusion planner
        # is not given the hurdles.
        self.disable_humanoid_collisions()
        unique_group = 0b0100
        collision_mask = 0xFFFF
        ray_collision_mask = 0b1011
        for hurdle in self.hurdles:
            self._p.setCollisionFilterGroupMask(hurdle.id, -1, unique_group, collision_mask)

        ray_from = [(xv[i, j], yv[i, j], 100.0) for i in range(nx) for j in range(ny)]
        ray_to = [(xv[i, j], yv[i, j], -100.0) for i in range(nx) for j in range(ny)]
        results = self._p.rayTestBatch(ray_from, ray_to, collisionFilterMask=ray_collision_mask)

        for idx, r in enumerate(results):
            i, j = idx // ny, idx % ny
            heightmap[i, j] = r[3][2]
            mask[i, j] = 0 if r[0] == -1 else 1

        self.enable_humanoid_collisions()
        return heightmap, mask, angle_to_wpt

    def disable_humanoid_collisions(self):
        """Moves the humanoid to a collision group that height rays ignore."""
        unique_group = 0b0100
        collision_mask = 0xFFFF
        robot_id = self.robot.object_id[0]
        self._p.setCollisionFilterGroupMask(robot_id, -1, unique_group, collision_mask)
        for body_part in self.robot.parts.values():
            self._p.setCollisionFilterGroupMask(
                robot_id, body_part.bodyPartIndex, unique_group, collision_mask
            )

    def enable_humanoid_collisions(self):
        robot_id = self.robot.object_id[0]
        self._p.setCollisionFilterPair(robot_id, robot_id, 4, 1, 0)
        for pair in self.robot.non_colliding_pairs:
            self._p.setCollisionFilterPair(robot_id, robot_id, pair[0], pair[1], 0)

    def get_height_from_coordinate(self, coords):
        """Ground height (first ray hit from above) at each (x, y) in `coords`."""
        ray_collision_mask = 0b1011
        heights = []
        collisions = []
        for cord in coords:
            result = self._p.rayTest(
                (cord[0], cord[1], 100.0), (cord[0], cord[1], -100.0), ray_collision_mask
            )[0]
            heights.append(result[3][2])
            collisions.append(0 if result[0] == -1 else 1)
        return heights, collisions

    # ---------------------------------------------------------------- waypoints

    def change_waypoint(self, waypoint):
        self.waypoint = waypoint

    def render_waypoint(self, w):
        if not hasattr(self, "waypoint_render"):
            self.waypoint_render = VSphere(self._p, radius=0.3, pos=None, rgba=[1, 1, 0, 1])
        self.waypoint_render.set_position(pos=w[:3])

    def next_stone_target(self, next_stones, k=4):
        """Renders the selected plan (only has an effect with render=True)."""
        self.mean_targets = next_stones[: k + 1]
        self.viz_k = k
        if self.is_render:
            self._handle_keyboard()
            self.camera.track(pos=self.robot.body_xyz)
            self._draw_plan()

    def _draw_plan(self):
        if self.viz_k is not None:
            for i in range(self.viz_k):
                self.all_targets[i].set_position(pos=self.mean_targets[i][:3])
                self.all_targets[i].set_color(DODGERBLUE)

    def create_target(self):
        # Visual-only spheres for the planned footsteps; called by EnvBase in render mode.
        self.all_targets = [VSphere(self._p, radius=0.15, pos=None) for _ in range(4)]

    # ----------------------------------------------------------- footstep logic

    def get_new_step_channel(self):
        return self.got_new_step

    def set_new_step_channel(self, flag):
        self.got_new_step = flag

    def get_foot_placement(self):
        return self.foot_placement, self.stepping_time_index

    def calc_potential(self):
        walk_target_theta = np.arctan2(
            self.walk_target[1] - self.robot.body_xyz[1],
            self.walk_target[0] - self.robot.body_xyz[0],
        )
        walk_target_delta = self.walk_target - self.robot.body_xyz
        self.angle_to_target = walk_target_theta - self.robot.body_rpy[2]
        self.distance_to_target = (walk_target_delta[0] ** 2 + walk_target_delta[1] ** 2) ** (1 / 2)
        self.linear_potential = -self.distance_to_target / self.scene.dt

    def calc_base_reward(self, action):
        old_linear_potential = self.linear_potential
        self.calc_potential()
        self.progress = self.linear_potential - old_linear_potential

        self.posture_penalty = 0
        if not -0.2 < self.robot.body_rpy[1] < 0.4:
            self.posture_penalty = abs(self.robot.body_rpy[1])
        if not -0.4 < self.robot.body_rpy[0] < 0.4:
            self.posture_penalty += abs(self.robot.body_rpy[0])

        self.speed_penalty = 0

        self.energy_penalty = self.electricity_cost * float(np.abs(action * self.robot.joint_speeds).mean())
        self.energy_penalty += self.stall_torque_cost * float(np.square(action).mean())
        self.joints_penalty = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        # Early termination: the pelvis drops too close to the lowest foot.
        height = self.robot.body_xyz[2] - np.min(self.robot.feet_xyz[:, 2])
        self.tall_bonus = 2.0 if height > self.done_height else -1.0
        self.done = self.done or self.tall_bonus < 0

    def calc_feet_state(self):
        self.target_reached = False

        self.foot_dist_to_target = np.sqrt(
            np.sum(
                (self.robot.feet_xyz[:, 0:2] - self.terrain_info[self.next_step_index, 0:2]) ** 2,
                axis=1,
            )
        )

        for i, f in enumerate(self.robot.feet):
            self.robot.feet_xyz[i] = f.pose().xyz()
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            self.robot.feet_contact[i] = 1.0 if contact_ids else 0.0
            if contact_ids and self.foot_dist_to_target[i] < 0.3:
                self.step_leg_idx = i
                self.target_reached = True

        if self.target_reached:
            self.target_reached_count += 1

            # Slight delay before advancing so the step bonus is not over-counted.
            if self.target_reached_count >= 2:
                self.set_new_step_channel(True)
                self.foot_placement.append(np.copy(self.robot.feet_xyz[self.swing_leg]))
                self.stepping_time_index.append(self.time_index)
                self.swing_leg = (self.swing_leg + 1) % 2
                self.next_step_index += 1
                np.random.rand()  # unused draw, kept to preserve the RNG stream
                self.target_reached_count = 0

            if self.next_step_index >= len(self.terrain_info):
                self.next_step_index -= 1

    def calc_step_reward(self):
        self.step_bonus = 0
        if self.target_reached and self.target_reached_count == 1:
            self.step_bonus = 50 * np.exp(-self.foot_dist_to_target.min() / 0.25)

        self.target_bonus = 0
        if self.next_step_index == len(self.terrain_info) - 1 and self.distance_to_target < 0.15:
            self.target_bonus = 2.0

    def calc_env_state(self, action):
        if not np.isfinite(self.robot_state).all():
            self.done = True

        cur_step_index = self.next_step_index
        self.calc_feet_state()
        self.calc_base_reward(action)
        self.calc_step_reward()
        self.targets = self.delta_to_k_targets(k=self.lookahead)

        if cur_step_index != self.next_step_index:
            self.targets = self.delta_to_k_targets(k=self.lookahead)
            self.calc_potential()

    def delta_to_k_targets(self, k=1):
        """Next k trajectory footsteps relative to the root (controller input format)."""
        targets = self.terrain_info[self.next_step_index : self.next_step_index + k]
        if len(targets) < k:
            targets = np.concatenate((targets, np.repeat(targets[[-1]], k - len(targets), axis=0)))
        self.walk_target = targets[[1], 0:3].mean(axis=0)

        deltas = targets[:, 0:3] - self.robot.body_xyz
        target_thetas = np.arctan2(deltas[:, 1], deltas[:, 0])
        angle_to_targets = target_thetas - self.robot.body_rpy[2]
        distance_to_targets = np.linalg.norm(deltas[:, 0:2], ord=2, axis=1)

        return np.stack(
            (
                np.sin(angle_to_targets) * distance_to_targets,  # x
                np.cos(angle_to_targets) * distance_to_targets,  # y
                deltas[:, 2],  # z
                targets[:, 4],  # x_tilt
                targets[:, 5],  # y_tilt
            ),
            axis=1,
        )
