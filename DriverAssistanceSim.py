# DriverAssistanceSim.py
import pybullet as p
import pybullet_data
import time
import numpy as np
import csv
import os
import math
import matplotlib.pyplot as plt
from PPO import PPOAgent

class DriverAssistanceSim:
    def __init__(self):
        self.timeStep = 0.1
        self.num_steps = 500
        self.max_range = 3.0
        self.z_offset = 0.05
        self.shelf_scale = 0.5
        self.num_cars = 15
        self.step_size = 0.1 # movement speed
        self.lane_width = 2.0
        self.offset = 0.1
        self.lane_width_total = 2 * self.lane_width + self.offset
        self.ang_vel = 5
        self.debug = False
        self.initialize_environment()

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Ground
        self.plane_id = p.loadURDF("plane.urdf")
        self.initialize_road()
        self.initialize_robot()

        # Respawn cars
        self.cars = []
        self.spawn_cars(self.num_cars)
        self.trajectory = []
        self.danger_points = []

    def initialize_environment(self):
        p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def initialize_road(self):
        self.road_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.road_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1]) 
        p.addUserDebugLine([0, self.lane_width - self.offset, 0.01], [200, self.lane_width - self.offset, 0.01], lineColorRGB=[1, 1, 1], lineWidth=4)
        p.addUserDebugLine([0, -self.lane_width + self.offset, 0.01], [200, -self.lane_width + self.offset, 0.01], lineColorRGB=[1, 1, 1], lineWidth=4)

        lane_length = 200
        lane_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01], rgbaColor=[0,0,0,1])
        lane_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01])
        _ = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=lane_collision, baseVisualShapeIndex=lane_visual, basePosition=[lane_length/2, 0, 0])

    def initialize_robot(self):
        y = np.random.uniform(-self.lane_width + (self.offset * 5), self.lane_width - (self.offset * 5))
        self.robot_start_pos = [0, y, 0.05]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot_path = os.path.join(pybullet_data.getDataPath(), "husky/husky.urdf")
        self.robot_id = p.loadURDF(robot_path, self.robot_start_pos, self.robot_start_orientation, globalScaling=1.0)  
        self.wheel_joints = [2, 3, 4, 5]

    def set_robot_wheel_velocities(self, linear_vel, angular_vel):
        L = 0.55
        v_left  = linear_vel - (angular_vel * L / 2)
        v_right = linear_vel + (angular_vel * L / 2)
        for i, joint in enumerate(self.wheel_joints):
            target = v_left if i % 2 == 0 else v_right
            p.setJointMotorControl2(bodyIndex=self.robot_id, jointIndex=joint, controlMode=p.VELOCITY_CONTROL, targetVelocity=target, force=100)

    def initialize_car(self, new_car=False):
        min_dist_between_cars = 1.0
        robot_safe_dist = 3.0
        placed = False
        attempts = 0
        while not placed and attempts < 500:
            attempts += 1
            x = np.random.uniform(0, 10)
            y = np.random.uniform(-self.lane_width + (self.offset * 2), self.lane_width - (self.offset * 2))
            dist_to_robot = np.sqrt((x - self.robot_start_pos[0])**2 + (y - self.robot_start_pos[1])**2)
            if dist_to_robot < robot_safe_dist:
                continue
            overlap = False
            for car in getattr(self, "cars", []):
                dist = np.sqrt((x - car['x_pos'])**2 + (y - car['y_pos'])**2)
                if dist < min_dist_between_cars:
                    overlap = True
                    break
            if not overlap:
                car_id = p.loadURDF("racecar/racecar.urdf", [x, y, 0], p.getQuaternionFromEuler([0,0,0]), globalScaling=1)
                velocity = np.random.uniform(-0.02, -0.08)
                self.cars.append({'id': car_id, 'vel': velocity, 'x_pos': x, 'y_pos': y})
                placed = True

    def get_robot_position(self, robot_id):
        return p.getBasePositionAndOrientation(robot_id)

    def clear_debug_lines(self):
        prev_ids = globals().get('debug_line_ids', [])
        for lid in prev_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        return []

    def get_angles(self, r_pos, z_offset, max_range):
        _, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_yaw = p.getEulerFromQuaternion(robot_orn)[2]
        angles = robot_yaw + np.linspace(-math.radians(30), math.radians(30), 15, endpoint=False)
        from_points, to_points = [], []
        for a in angles:
            start = [r_pos[0], r_pos[1], r_pos[2]]
            end = [r_pos[0] + max_range * math.cos(a), r_pos[1] + max_range * math.sin(a), r_pos[2]]
            from_points.append(start)
            to_points.append(end)
        return angles, from_points, to_points

    def get_closest_hit(self, hits_cube):
        if len(hits_cube) > 0:
            hits_cube.sort(key=lambda x: x[0])
            closest_hit = hits_cube[0][1]
            angle = hits_cube[0][2]
            return closest_hit[0], closest_hit[1], angle
        else:
            return 0, 0, 0

    def calculate_euclidean_dist(self, hit_position, r_pos, z_offset):
        return math.sqrt((hit_position[0] - r_pos[0])**2 + (hit_position[1] - r_pos[1])**2 + (hit_position[2] - (r_pos[2] + z_offset))**2)

    def move_cars(self):
        for car in self.cars:
            cid = car['id']
            pos, orn = p.getBasePositionAndOrientation(cid)
            new_pos = [pos[0] - car['vel'], pos[1], pos[2]]
            p.resetBasePositionAndOrientation(cid, new_pos, orn)

    def get_camera_image(self, robot_id, direction, yaw_offset_deg=90):
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2] + math.radians(yaw_offset_deg)
        pitch = -15 * math.pi/180
        cam_offset = [0.5, 0.0, 0.3]
        if direction == "left":
            cam_offset = [0.5, 0.3, 0.3]
        elif direction == "right":
            cam_offset = [0.5, -0.3, 0.3]
        cam_pos = [pos[0] + cam_offset[0], pos[1] + cam_offset[1], pos[2] + cam_offset[2]]
        cam_target = [cam_pos[0] + math.cos(yaw) * math.cos(pitch), cam_pos[1] + math.sin(yaw) * math.cos(pitch), cam_pos[2] + math.sin(pitch)]
        up = [0, 0, 1]
        view = p.computeViewMatrix(cam_pos, cam_target, up)
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.012, farVal=3.0)
        width, height, rgb, depth, seg = p.getCameraImage(width=128, height=128, viewMatrix=view, projectionMatrix=proj)
        return rgb, depth, seg

    def four_parallel_robot_beams(self, robot_pos, max_range=None):
        if max_range is None:
            max_range = self.max_range
        BEAM_Z = 0.1
        GAP = 0.1
        ANG = math.radians(10)
        base = [robot_pos[0], robot_pos[1], BEAM_Z]
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]
        fx = math.cos(yaw); fy = math.sin(yaw)
        px = -math.sin(yaw); py = math.cos(yaw)
        left_start  = [base[0] + px * (-GAP/2), base[1] + py * (-GAP/2), BEAM_Z]
        right_start = [base[0] + px * (+GAP/2), base[1] + py * (+GAP/2), BEAM_Z]
        fx_L = math.cos(yaw + ANG); fy_L = math.sin(yaw + ANG)
        fx_R = math.cos(yaw - ANG); fy_R = math.sin(yaw - ANG)
        starts = [left_start, right_start, base, base]
        ends = [
            [left_start[0]  + fx * max_range, left_start[1]  + fy * max_range, BEAM_Z],
            [right_start[0] + fx * max_range, right_start[1] + fy * max_range, BEAM_Z],
            [base[0] + fx_L * max_range, base[1] + fy_L * max_range, BEAM_Z],
            [base[0] + fx_R * max_range, base[1] + fy_R * max_range, BEAM_Z]
        ]
        results = p.rayTestBatch(starts, ends)
        car_ids = [c['id'] for c in getattr(self, "cars", [])]
        distances = []
        debug_ids = self.clear_debug_lines()
        for i, res in enumerate(results):
            hit = res[0]
            frac = res[2]
            dist = frac * max_range if hit in car_ids else max_range
            distances.append(dist)
            endp = [starts[i][0] + (ends[i][0] - starts[i][0]) * frac, starts[i][1] + (ends[i][1] - starts[i][1]) * frac, BEAM_Z]
            color = [1,1,1] if dist < max_range else [0,1,1]
            lid = p.addUserDebugLine(starts[i], endp, color, 2 if i < 2 else 1, 0)
            debug_ids.append(lid)
        globals()['debug_line_ids'] = debug_ids
        return distances[0], distances[1], distances[2], distances[3]

    def beam_sensor(self, robot_pos, z_offset=0.1, max_range=5):
        r_pos = [robot_pos[0], robot_pos[1], z_offset]
        new_debug_ids = self.clear_debug_lines()
        angles, from_points, to_points = self.get_angles(r_pos, z_offset, max_range)
        results = p.rayTestBatch(from_points, to_points)
        car_ids = [car['id'] for car in getattr(self, "cars", [])]
        hits_cube = []
        for i, res in enumerate(results):
            hit_object_uid = res[0]
            hit_fraction = res[2]
            hit_position = res[3]
            if hit_object_uid in car_ids:
                ang_w = angles[i]
                measured = hit_fraction * max_range
                ep = [r_pos[0] + measured * math.cos(ang_w), r_pos[1] + measured * math.sin(ang_w), r_pos[2] + z_offset]
                lid = p.addUserDebugLine(from_points[i], ep, lineColorRGB=[1,0,0], lineWidth=2.0, lifeTime=0)
                new_debug_ids.append(lid)
                dist = self.calculate_euclidean_dist(hit_position, r_pos, z_offset)
                hits_cube.append((dist, hit_position, ang_w))
        globals()['debug_line_ids'] = new_debug_ids
        return self.get_closest_hit(hits_cube)

    def get_state(self, robot_pos):
        b1_left, b2_right, b3_angL, b4_angR = self.four_parallel_robot_beams(robot_pos)
        b1 = b1_left / self.max_range
        b2 = b2_right / self.max_range
        b3 = b3_angL / self.max_range
        b4 = b4_angR / self.max_range
        hit_x, hit_y, angle = self.beam_sensor(robot_pos, z_offset=self.z_offset, max_range=self.max_range)
        if abs(angle) <= math.radians(30):
            if hit_x == 0 and hit_y == 0:
                beam_dist = self.max_range
            else:
                beam_dist = np.linalg.norm(np.array([hit_x, hit_y]) - np.array(robot_pos[:2]))
        else:
            beam_dist = self.max_range
        beam_dist = beam_dist / self.max_range
        left_closeness  = (1 - beam_dist) * max(0,  angle)
        right_closeness = (1 - beam_dist) * max(0, -angle)
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lane_offset = pos[1] / (self.lane_width - 0.2)
        yaw = p.getEulerFromQuaternion(orn)[2]
        heading_error = yaw / np.pi
        return np.array([b1,b2,b3,b4,beam_dist, left_closeness, right_closeness, lane_offset, heading_error], dtype=np.float32)

    def compute_reward(self, state, action, done):
        # state indices:
        # 0=b1,1=b2,2=b3,3=b4,4=beam_dist,5=left_close,6=right_close,7=lane_offset,8=heading
        b1, b2, b3, b4 = state[0], state[1], state[2], state[3]
        beam = state[4]
        left_score = state[5]
        right_score = state[6]
        left_clear  = (b1 + b3) * 0.5
        right_clear = (b2 + b4) * 0.5
        front_clear = beam
        reward = 0.0

        # smoothing parameters
        SWITCH_PENALTY = 0.12   # small penalty to discourage micro-switching but allow turns
        FRONT_SAFE_BONUS = 0.08
        ANGLE_UTIL_BONUS = 0.4

        # 1) Basic forward / front reward (small)
        if action == 0:
            reward += FRONT_SAFE_BONUS * front_clear

        # 2) directional clarity (all actions get it)
        reward += 0.4 * b1 + 0.4 * b2 + 0.25 * b3 + 0.25 * b4

        # 3) Penalize being too close laterally (but modestly)
        reward -= 0.9 * (left_score ** 1.1)
        reward -= 0.9 * (right_score ** 1.1)

        # 4) Encourage useful turns (only when they open up space)
        # if left side is blocked and right is clearer -> prefer turning right (action=2)
        if (right_clear - left_clear) > 0.05 and action == 2:
            reward += 0.8 * (right_clear - left_clear)
        if (left_clear - right_clear) > 0.05 and action == 1:
            reward += 0.8 * (left_clear - right_clear)

        # 5) If front is very close, prefer turning away from denser side
        if front_clear < 0.45:
            if left_clear > right_clear and action == 1:
                reward += 1.0 * (left_clear - right_clear)
            if right_clear > left_clear and action == 2:
                reward += 1.0 * (right_clear - left_clear)

        # 6) Use angled beams
        reward += ANGLE_UTIL_BONUS * (b3 + b4)

        # 7) Lane keeping: small reward for being near center; stronger reward for corrective turn
        lane_offset = state[7]  # normalized -1..1
        if abs(lane_offset) < 0.8:
            reward += 0.05 * (1 - abs(lane_offset))
        if lane_offset > 0.6 and action == 2:
            reward += 0.6
        if lane_offset < -0.6 and action == 1:
            reward += 0.6

        # 8) Done/Collision
        if done:
            reward -= 40.0

        # 9) Switching penalty (small)
        if not hasattr(self, "prev_action"):
            self.prev_action = action
        if action != self.prev_action:
            reward -= SWITCH_PENALTY
        self.prev_action = action

        return float(np.clip(reward, -12.0, 12.0))

    def spawn_cars(self, car_count):
        for _ in range(car_count):
            self.initialize_car(new_car=True)

    def plot_trajectory(self, ep, steps, reward, loss):
        if len(self.trajectory) == 0:
            return
        plt.figure(figsize=(10, 4))
        traj = np.array([(x, y) for (x, y, a) in self.trajectory])
        plt.plot(traj[:, 0], traj[:, 1], label="Full Trajectory")
        colors = {0: "blue", 1: "green", 2: "red"}
        labels = {0: "Straight", 1: "Left", 2: "Right"}
        for x, y, a in self.trajectory:
            plt.scatter(x, y, color=colors[a], s=8)
        for x, y, a in self.danger_points:
            plt.scatter(x, y, color=colors[a], s=60, marker='X', edgecolors='black')
        plt.axhline(self.lane_width, color='black', linestyle='--')
        plt.axhline(-self.lane_width, color='black', linestyle='--')
        plt.title(f"Robot Trajectory – Episode: {ep}, Steps: {steps}, Reward: {reward:.2f}, Loss: {loss if type(loss) in [float, int] else ''}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        for a in [0, 1, 2]:
            plt.scatter([], [], c=colors[a], label=labels[a])
        plt.legend()
        filename = f"trajectory_episode_{ep}.png"
        plt.savefig(filename, dpi=200)
        plt.close()
    
    def plot_rewards(self,rewards, window=10):
        """
        Plot reward per episode and moving average.
        :param rewards: list of episode rewards
        :param window: window size for moving average
        """
        if len(rewards) < window:
            print("Window is larger than reward list length!")
            return
        
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(12, 5))
        plt.plot(rewards, alpha=0.35, label='Reward per Episode')
        plt.plot(range(window-1, len(rewards)), moving_avg, linewidth=2.5, label=f'Moving Average ({window})')
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward Trend")
        plt.legend()
        plt.grid()
        plt.savefig("reward_plot", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_steps(self,steps, window=10):
        """
        Plot reward per episode and moving average.
        :param rewards: list of episode rewards
        :param window: window size for moving average
        """
        if len(steps) < window:
            print("Window is larger than reward list length!")
            return
        
        moving_avg = np.convolve(steps, np.ones(window)/window, mode='valid')

        plt.figure(figsize=(12, 5))
        plt.plot(steps, alpha=0.35, label='steps per Episode')
        plt.plot(range(window-1, len(steps)), moving_avg, linewidth=2.5, label=f'Moving Average ({window})')
        plt.xlabel("Episode")
        plt.ylabel("steps")
        plt.title("steps Trend")
        plt.legend()
        plt.grid()
        plt.savefig("steps_plot", dpi=300, bbox_inches='tight')
        plt.show()

    def run(self):
        agent = PPOAgent(state_dim=9, action_dim=3)
        checkpoint_file = "ppo_checkpoint.pth"
        if os.path.exists(checkpoint_file):
            agent.load_model(checkpoint_file)
            print("Loaded previous PPO checkpoint — training will continue ✔")
        else:
            print("No checkpoint found — starting fresh ❗")

        EPISODES = 500
        reward_list = []
        step_list=[]
        # small epsilon-greedy heuristic probability for early exploration (optional)
        heuristic_epsilon = 0.0  # set e.g. 0.15 to occasionally use heuristic decisions

        for ep in range(EPISODES):
            print("\n==== EPISODE", ep, "====")
            episode_reward = 0
            self.reset()
            terminal_step = self.num_steps
            reset_angle = 0

            for step in range(self.num_steps):
                action_method = ""
                if step % 100 == 0 and step > 0:
                    self.spawn_cars(2)
                p.stepSimulation()
                # time.sleep(0.01)  # speed up in DIRECT mode by lowering sleep
                self.move_cars()

                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                robot_pos = [pos[0], pos[1], pos[2]]

                # STATE
                state = self.get_state(robot_pos)

                # ACTION (agent)
                use_heuristic = (np.random.rand() < heuristic_epsilon)
                if use_heuristic:
                    # lightweight heuristic for exploration (same as before but simpler)
                    b1, b2, b3, b4, front, left_close, right_close, _, _ = state
                    left_block  = (b1 < 0.6) or (b3 < 0.6) or (left_close > 0.25)
                    right_block = (b2 < 0.6) or (b4 < 0.6) or (right_close > 0.25)
                    front_block = (front < 0.55) or ((b1 < 0.55) or (b2 < 0.55))
                    if front_block:
                        if left_block and not right_block:
                            action = 2
                        elif right_block and not left_block:
                            action = 1
                        else:
                            action = 1 if left_close < right_close else 2
                    else:
                        if left_block and not right_block:
                            action = 2
                        elif right_block and not left_block:
                            action = 1
                        else:
                            action = 0
                    logprob = 0.0
                    action_method = "action_rules"

                else:
                    action, logprob = agent.act(state)
                    action_method = "PPO_rules"

                self.trajectory.append((robot_pos[0], robot_pos[1], action))
                beam = state[0]
                if beam < 1.0:
                    self.danger_points.append((robot_pos[0], robot_pos[1], action))

                # Map actions to wheel speeds
                if action == 0:
                    self.set_robot_wheel_velocities(4, 0.0)
                elif action == 1:
                    self.set_robot_wheel_velocities(3, +self.ang_vel)
                elif action == 2:
                    self.set_robot_wheel_velocities(3, -self.ang_vel)

                p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=0, cameraPitch=-80, cameraTargetPosition=robot_pos)

                # Terminal condition
                done = ((state[4] < (0.5/self.max_range)) or (abs(pos[1]) > (self.lane_width + self.offset)))

                # REWARD
                reward = self.compute_reward(state, action, done)
                episode_reward += reward

                print(f"Episode={ep} Step={step} Action={action} Reward={reward:.3f} Total={episode_reward:.3f} action_method={action_method}")

                agent.remember(state, action, logprob, reward, done)

                if done:
                    terminal_step = step
                    print("Episode terminated early at step", step)
                    break

            step_list.append(terminal_step)
            reward_list.append(episode_reward)
            print("Episode Reward =", episode_reward, "  steps =",terminal_step)

            # Train PPO after each episode (only if memory has enough samples per PPO.min_batch_size)
            loss = agent.train()
            self.plot_trajectory(ep, terminal_step, episode_reward, loss)

        print("All Episodes Completed: reward list - ", reward_list)
        self.plot_rewards(reward_list,20)
        self.plot_steps(step_list,20)
        p.disconnect()

if __name__ == "__main__":
    sim = DriverAssistanceSim()
    sim.run()
