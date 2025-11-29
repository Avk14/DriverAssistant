import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from PPO import PPOAgent

class DriverAssistanceSim:
    def __init__(self):
        self.timeStep = 0.1
        self.num_steps = 500
        self.max_range = 5
        self.z_offset = 0.05
        self.shelf_scale = 0.5
        self.num_cars = 10
        self.step_size = 0.1 # movement speed
        self.lane_width = 2.0
        self.offset = 0.1
        self.lane_width_total = 2 * self.lane_width + self.offset
        self.ang_vel = 5
        self.initialize_environment()
        # self.initialize_road()
        # self.initialize_robot()
    
    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Ground
        self.plane_id = p.loadURDF("plane.urdf")

        # Recreate your road using existing function
        self.initialize_road()

        # Reset robot to original position
        self.initialize_robot()

        # Respawn cars
        self.cars = []
        self.spawn_cars(self.num_cars)
        self.trajectory = []
        self.danger_points = []

    def initialize_environment(self):
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.timeStep)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

    def initialize_road(self):
        self.road_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(self.road_id, -1, rgbaColor=[0.5, 0.5, 0.5, 1]) 
        # Draw lane lines
        p.addUserDebugLine([0, self.lane_width - self.offset, 0.01], [200, self.lane_width - self.offset, 0.01], lineColorRGB=[1, 1, 1], lineWidth=4)
        p.addUserDebugLine([0, -self.lane_width + self.offset, 0.01], [200, -self.lane_width + self.offset, 0.01], lineColorRGB=[1, 1, 1], lineWidth=4)

        lane_length = 200
        lane_color = [0, 0, 0, 1]  

        lane_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01], rgbaColor=lane_color)
        lane_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[lane_length / 2, self.lane_width_total / 2, 0.01])
        lane_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=lane_collision, baseVisualShapeIndex=lane_visual, basePosition=[lane_length/2, 0, 0])

    def initialize_robot(self):
        y = np.random.uniform(-self.lane_width + (self.offset * 5), self.lane_width - (self.offset * 5))
        self.robot_start_pos = [0, y, 0.05]
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot_path = os.path.join(pybullet_data.getDataPath(), "husky/husky.urdf")
        self.robot_id = p.loadURDF(robot_path, self.robot_start_pos, self.robot_start_orientation, globalScaling=1.0)  
        self.wheel_joints = [2, 3, 4, 5] 

    def set_robot_wheel_velocities(self, linear_vel, angular_vel):
        L = 0.55  # Husky wheel base (distance between left/right wheels)
        
        # Convert linear+angular to left/right wheel velocities
        v_left  = linear_vel - (angular_vel * L / 2)
        v_right = linear_vel + (angular_vel * L / 2)
        
        for i, joint in enumerate(self.wheel_joints):
            if i % 2 == 0:  # left wheels
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=v_left,
                    force=100
                )
            else:   # right wheels
                p.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=v_right,
                    force=100
                )
    
    def initialize_car(self, new_car=False):
        min_dist_between_cars = 1.0   # minimum distance between cars
        robot_safe_dist = 3.0         # minimum distance from robot start

        placed = False
        while not placed:
            x = np.random.uniform(0, 10)
            y = np.random.uniform(-self.lane_width + (self.offset * 2), self.lane_width - (self.offset * 2))
            
            # Check distance to robot
            dist_to_robot = np.sqrt((x - self.robot_start_pos[0])**2 + (y - self.robot_start_pos[1])**2)
            if dist_to_robot < robot_safe_dist:
                continue  # too close to robot, try again

            # Check distance to other cars
            overlap = False
            for car in self.cars:
                dist = np.sqrt((x - car['x_pos'])**2 + (y - car['y_pos'])**2)
                if dist < min_dist_between_cars:
                    overlap = True
                    break
            
            if not overlap:
                car_id = p.loadURDF("racecar/racecar.urdf", [x, y, 0],
                                    p.getQuaternionFromEuler([0,0,0]),
                                    globalScaling=1)
                velocity = np.random.uniform(-0.02, -0.08)
                self.cars.append({'id': car_id, 'vel': velocity, 'x_pos': x, 'y_pos': y})
                placed = True

    def get_robot_position(self, robot_id):
        return p.getBasePositionAndOrientation(robot_id)

    def clear_debug_lines(self):
        # clear old debug lines
        prev_ids = globals().get('debug_line_ids', [])
        for lid in prev_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        return []
    
    def get_debug_lines(self):
        return globals().get('debug_line_ids', [])

    def get_angles(self, r_pos, z_offset, max_range ):
        angles = np.linspace(-math.pi/3, math.pi/3, 30, endpoint=False)
        from_points = []
        to_points = []
        for a in angles:
            ang_w = a
            start = [r_pos[0], r_pos[1], r_pos[2] ]
            end = [r_pos[0] + max_range * math.cos(ang_w),
                r_pos[1] + max_range * math.sin(ang_w),
                r_pos[2]]
            from_points.append(start)
            to_points.append(end)
        return angles, from_points, to_points
    
    def get_closest_hit(self, hits_cube):
        # return the closest hit
        # dist, hit_pos, ang_w = hits_cube
        if len(hits_cube) > 0:
            hits_cube.sort(key=lambda x: x[0])  # sort by distance
            closest_hit = hits_cube[0][1]
            angle = hits_cube[0][2]
            return closest_hit[0], closest_hit[1], angle
        else:
            return 0, 0, 0  # no hit
    
    def calculate_euclidean_dist(self, hit_position, r_pos, z_offset):
        return math.sqrt(
            (hit_position[0] - r_pos[0])**2 +
            (hit_position[1] - r_pos[1])**2 +
            (hit_position[2] - (r_pos[2] + z_offset))**2
        )
        
    def move_cars(self):
        for car in self.cars:
            cid = car['id']
            pos, orn = p.getBasePositionAndOrientation(cid)
            new_pos = [pos[0] - car['vel'], pos[1], pos[2]]  # move along -x
            p.resetBasePositionAndOrientation(cid, new_pos, orn)

    def get_camera_image(self, robot_id, direction, yaw_offset_deg=90):
        """
        Capture a camera image looking at an offset angle from robot's forward.
        yaw_offset_deg:
            0 -> forward
            90 -> left
            -90 -> right
        """
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        # Convert robot quaternion to Euler
        euler = p.getEulerFromQuaternion(orn)
        
        # Apply yaw offset
        yaw = euler[2] + math.radians(yaw_offset_deg)
        pitch = -15 * math.pi/180  # slightly down

        if direction == "left":
            cam_offset = [0.5, 0.3, 0.3]  # Adjusted for better view
        elif direction == "right":
            cam_offset = [0.5, -0.3, 0.3]  # Adjusted for better view
        else:
            cam_offset = [0.5, 0.0, 0.3]  # Forward view
        
        cam_pos = [pos[0] + cam_offset[0],
           pos[1] + cam_offset[1],
           pos[2] + cam_offset[2]]

        # Compute camera target from yaw/pitch
        cam_target = [
            cam_pos[0] + math.cos(yaw) * math.cos(pitch),
            cam_pos[1] + math.sin(yaw) * math.cos(pitch),
            cam_pos[2] + math.sin(pitch)
        ]
        
        up = [0, 0, 1]  # world-up
        view = p.computeViewMatrix(cam_pos, cam_target, up)
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.012, farVal=3.0)
        
        width, height, rgb, depth, seg = p.getCameraImage(
            width=128, height=128,
            viewMatrix=view, projectionMatrix=proj
        )
        
        return rgb, depth, seg

    def detect_lane_free_from_camera(self, rgb_img):
        # Reshape image (128x128)
        img = np.reshape(rgb_img, (128, 128, 4))[:, :, :3]  # drop alpha
        gray = img.mean(axis=2)  # simple grayscale

        # Divide image into 3 zones: left, center, right
        left_zone   = gray[:, :40]
        center_zone = gray[:, 40:88]
        right_zone  = gray[:, 88:]

        # Simple threshold: dark = obstacle
        left_blocked  = np.mean(left_zone) < 80
        right_blocked = np.mean(right_zone) < 80

        return {"left": not left_blocked, "right": not right_blocked}
    
    def show_camera_image(self, rgb):
        img = np.reshape(rgb, (128, 128, 4))[:, :, :3]
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def four_parallel_robot_beams(self, robot_pos, max_range=None):
        """
        4 robot-aligned beams:
        - 2 forward inside robot width
        - 2 angled beams (±20°)
        Returns meters: B1_left, B2_right, B3_ang_left, B4_ang_right
        """
        if max_range is None:
            max_range = self.max_range

        BEAM_Z = 0.08
        GAP = 0.38          # inside Husky width (0.55 m)
        ANG = math.radians(20)  # 20° for angled beams

        base = [robot_pos[0], robot_pos[1], BEAM_Z]
        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        yaw = p.getEulerFromQuaternion(orn)[2]

        fx = math.cos(yaw);  fy = math.sin(yaw)
        px = -math.sin(yaw); py = math.cos(yaw)

        # ---- 2 inside-width beam start points ----
        left_start  = [base[0] + px * (-GAP/2), base[1] + py * (-GAP/2), BEAM_Z]
        right_start = [base[0] + px * (+GAP/2), base[1] + py * (+GAP/2), BEAM_Z]

        # ---- 2 angled directions ----
        fx_L = math.cos(yaw + ANG);  fy_L = math.sin(yaw + ANG)
        fx_R = math.cos(yaw - ANG);  fy_R = math.sin(yaw - ANG)

        starts = [
            left_start, right_start, base, base
        ]
        ends = [
            [left_start[0]  + fx * max_range, left_start[1]  + fy * max_range, BEAM_Z],   # B1 center-left
            [right_start[0] + fx * max_range, right_start[1] + fy * max_range, BEAM_Z],   # B2 center-right
            [base[0]        + fx_L * max_range, base[1]        + fy_L * max_range, BEAM_Z], # B3 angled-left
            [base[0]        + fx_R * max_range, base[1]        + fy_R * max_range, BEAM_Z]  # B4 angled-right
        ]

        results = p.rayTestBatch(starts, ends)
        car_ids = [c['id'] for c in self.cars]

        distances = []
        debug_ids = self.clear_debug_lines()

        for i, res in enumerate(results):
            hit = res[0]
            frac = res[2]
            dist = frac * max_range if hit in car_ids else max_range
            distances.append(dist)

            endp = [
                starts[i][0] + (ends[i][0] - starts[i][0]) * frac,
                starts[i][1] + (ends[i][1] - starts[i][1]) * frac,
                BEAM_Z
            ]
            color = [1,0,1] if dist < max_range else [0,1,1]
            lid = p.addUserDebugLine(starts[i], endp, color, 2 if i < 2 else 1, 0)
            debug_ids.append(lid)

        globals()['debug_line_ids'] = debug_ids
        return distances[0], distances[1], distances[2], distances[3]



    def beam_sensor(self, robot_pos, z_offset=0.1, max_range=5):
        r_pos = [robot_pos[0], robot_pos[1], z_offset]

        new_debug_ids = self.clear_debug_lines()
        angles, from_points, to_points = self.get_angles(r_pos, z_offset, max_range)

        results = p.rayTestBatch(from_points, to_points)
        car_ids = [car['id'] for car in self.cars]
        hits_cube = []
        for i, res in enumerate(results):
            hit_object_uid = res[0] # object unique id of the hit object
            hit_fraction = res[2] # hit fraction along the ray in range [0,1] along the ray
            hit_position = res[3] # hit position in Cartesian world coordinates

            if hit_object_uid in car_ids:
                color = [1,0,0]
                ang_w = angles[i]
                # print("ang_w", ang_w)
                measured = hit_fraction * max_range
                ep = [r_pos[0] + measured * math.cos(ang_w),
                    r_pos[1] + measured * math.sin(ang_w),
                    r_pos[2] + z_offset]
            
                lid = p.addUserDebugLine(from_points[i], ep, lineColorRGB=color, lineWidth=2.0, lifeTime=0)
                new_debug_ids.append(lid)

                # find euclidean distance
                dist = self.calculate_euclidean_dist(hit_position, r_pos, z_offset)

                # print("hittttttt uid = ", hit_object_uid, " at fraction ", hit_fraction, " at pos ", hit_position, " dist ", dist)

                hits_cube.append((dist, hit_position, ang_w))
            else:
                color = [0,1,0]
                # lid = p.addUserDebugLine(from_points[i], to_points[i], lineColorRGB=color, lineWidth=1.0, lifeTime=0)
                # new_debug_ids.append(lid)

            globals()['debug_line_ids'] = new_debug_ids
        return self.get_closest_hit(hits_cube)

    def is_path_clear(self, rgb_img, threshold=80):
        # Convert flat list to (H, W, 3) array
        img = np.reshape(rgb_img, (128, 128, 4))[:, :, :3]  # drop alpha
        gray = img.mean(axis=2)  # grayscale
        # print("value is ",np.mean(gray) )
        return np.mean(gray) > threshold  # True = path is clear

    def get_state(self, robot_pos):

        b1_left, b2_right, b3_angL, b4_angR = self.four_parallel_robot_beams(robot_pos)

        # normalize 0..1
        b1 = b1_left / self.max_range
        b2 = b2_right / self.max_range
        b3 = b3_angL / self.max_range
        b4 = b4_angR / self.max_range


        # Beam-based obstacle detection
        hit_x, hit_y, angle = self.beam_sensor(robot_pos, z_offset=self.z_offset, max_range=self.max_range)
        if hit_x == 0 and hit_y == 0:
            beam_dist = self.max_range
        else:
            beam_dist = np.linalg.norm(np.array([hit_x, hit_y]) - np.array(robot_pos[:2]))
        beam_dist = beam_dist / self.max_range   # normalize 0–1

        # Camera clarity scores
        rgb_left, _, _ = self.get_camera_image(self.robot_id, "left", yaw_offset_deg=90)
        rgb_right, _, _ = self.get_camera_image(self.robot_id, "right", yaw_offset_deg=-90)
        imgL = np.reshape(rgb_left, (128,128,4))[:, :, :3].mean() / 255.0
        imgR = np.reshape(rgb_right, (128,128,4))[:, :, :3].mean() / 255.0

        left_score = float(imgL)
        right_score = float(imgR)

        # Lane offset
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lane_offset = pos[1] / (self.lane_width - self.offset)     # normalized -1 to +1

        # Heading error (yaw)
        yaw = p.getEulerFromQuaternion(orn)[2]
        heading_error = yaw / np.pi   # normalize -1 to 1

        return np.array([b1,b2,b3,b4,beam_dist, left_score, right_score, lane_offset, heading_error], dtype=np.float32)


    def compute_reward(self, state, action):
        beam = state[4]
        left_str_beam = state[0]
        right_str_beam = state[1]
        left_ang = state[2]
        right_ang = state[3]

        reward = 0.0

        if beam < (1.0/self.max_range):
            reward -= 5  # obstacle too near
        if left_str_beam == 1 and right_str_beam== 1 and action==0:
            reward+=4
        elif left_ang==1 and left_str_beam==1 and action==1:
            reward+=4   
        elif right_ang==1 and right_str_beam==1 and action==2:
            reward+=4

        # if action == 1 and left:
        #     reward += 1  # good left turn
        # if action == 2 and right:
        #     reward += 1  # good right turn

        if action == 0:
            reward += 0.5  # small reward for going straight

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        if abs(pos[1]) > self.lane_width:  # robot outside lane
            reward -= 5
            print("Lane crossed! Penalty applied")

        return reward
    
    def spawn_cars(self, car_count):
        for _ in range(car_count):
            self.initialize_car(new_car=True)
    
    def plot_trajectory(self, ep, steps, reward, loss):
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

        plt.title(f"Robot Trajectory with Actions – Episode: {ep}, Steps: {steps}, Reward: {reward:.2f}, Loss: {loss:.4f}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        for a in [0, 1, 2]:
            plt.scatter([], [], c=colors[a], label=labels[a])

        plt.legend()
        filename = f"trajectory_episode_{ep}.png"
        plt.savefig(filename, dpi=200)
        plt.close()

    def run(self):
        agent = PPOAgent()
        EPISODES = 30
        reward_list = []

        for ep in range(EPISODES):
            print("\n==== EPISODE", ep, "====")
            episode_reward = 0
            self.reset()
            terminal_step = self.num_steps
            reset_angle = 0

            for step in range(self.num_steps):
                if step % 100 == 0 and step > 0:
                    self.spawn_cars(3) # Respawn cars
                p.stepSimulation()
                time.sleep(self.timeStep)
                self.move_cars()

                pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                robot_pos = [pos[0], pos[1], pos[2]]

                # -------------------
                # STATE
                # -------------------
                # Step 1. Get state from environment
                state = self.get_state(robot_pos)
                # -------------------
                # PPO ACTION
                # -------------------
                # Step 2. Get action from PPO agent, by passing state to policy network
                action, logprob = agent.act(state)

                self.trajectory.append((robot_pos[0], robot_pos[1], action))
                beam = state[0]
                if beam < 1.0:
                    self.danger_points.append((robot_pos[0], robot_pos[1], action))

                # Map actions to wheel speeds
                if action == 0: # go straight 
                    self.set_robot_wheel_velocities(7, reset_angle)
                    reset_angle = 0  # reset angle for straight
                elif action == 1: # turn left
                    self.set_robot_wheel_velocities(3, +self.ang_vel)
                    reset_angle -= self.ang_vel
                    # if reset_angle < -math.pi:
                    #     reset_angle += 2 * math.pi 
                elif action == 2: # turn right
                    self.set_robot_wheel_velocities(3, -self.ang_vel)
                    reset_angle += self.ang_vel
                
                p.resetDebugVisualizerCamera(
                    cameraDistance=5,
                    cameraYaw=0,
                    cameraPitch=-80,
                    cameraTargetPosition=robot_pos  # camera follows robot
                )

                # Step 3. Compute reward
                reward = self.compute_reward(state, action)
                episode_reward += reward

                print("Episode = ",ep , "Step =", step, " State =", state, " Action =", action, " LogProb =", logprob, " Reward =", reward, "total_reward =", episode_reward)

                # Terminal condition
                done = ((state[4] < (0.5/self.max_range)) or (abs(pos[1]) > (self.lane_width + self.offset)))

                if done:
                    reward-=100
                # Step 4. Store (s, a, logprob, r, done) in PPO memory
                agent.remember(state, action, logprob, reward, done)

                if done:
                    terminal_step = step
                    print("Episode terminated early")
                    break
            reward_list.append(episode_reward)
            print("Episode Reward =", episode_reward)

            # Step 5. Collect the above batch data, and train PPO agent after each episode using value function network
            loss = agent.train()
            self.plot_trajectory(ep, terminal_step, episode_reward, loss)

        print("All Episodes Completed: reward list - ", reward_list)
        p.disconnect()

sim = DriverAssistanceSim()
sim.run()