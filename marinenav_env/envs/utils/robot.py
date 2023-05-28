import numpy as np
import copy



class Perception:

    def __init__(self,cooperative:bool=False):
        # 2D LiDAR model with detection area as a sector
        self.range = 15.0 # range of beams (meter)
        self.angle = 2 * np.pi # detection angle range
        self.len_obs_history = 1 # the window size of observation history
        self.observation_format(cooperative)
        self.observed_obs = [] # indices of observed static obstacles
        self.observed_objs = [] # indiced of observed dynamic objects

    def observation_format(self,cooperative:bool=False):
        # format: {"self": [velocity,goal], 
        #          "static":[[obs_1.x,obs_1.y,obs_1.r],...,[obs_n.x,obs_n.y,obs_n.r]],
        #          "dynamic":{id_1:[[robot_1.x,robot_1.y,robot_1.vx,robot_1.vy]_(t-m),...,[]_t]...}
        if cooperative:
            self.observation = dict(self=[],static=[],dynamic={})
        else:
            self.observation = dict(self=[],static=[])

class Odometry:
    def __int__(self):
        self.sigma_g = 5/360 * np.pi #gyro noise
        self.sigma_a = 0.05#accel noise
        self.sigma_a_p = 0.05#accel noise percentage

    def 


class Robot:

    def __init__(self,cooperative:bool=False):
        
        # parameter initialization
        self.cooperative = cooperative # if the robot is cooperative or not
        self.dt = 0.2 # discretized time step (second)
        self.N = 5 # number of time step per action
        self.perception = Perception(cooperative)
        self.length = 1.0 
        self.width = 0.5
        self.r = 0.8 # collision range
        self.detect_r = 0.5*np.sqrt(self.length**2+self.width**2) # detection range
        self.goal_dis = 2.0 # max distance to goal considered as reached   
        self.max_speed = 2.0
        self.a = np.array([-0.4,0.0,0.4]) # linear accelerations (m/s^2)
        self.w = np.array([-np.pi/6,0.0,np.pi/6]) # angular velocities (rad/s)
        self.compute_k() # cofficient of water resistance
        self.compute_actions() # list of actions

        self.x = None # x coordinate
        self.y = None # y coordinate
        self.theta = None # steering heading angle
        self.speed = None # steering foward speed
        self.velocity = None # velocity wrt sea floor
        
        self.start = None # start position
        self.goal = None # goal position
        self.reach_goal = False

        self.init_theta = 0.0 # theta at initial position
        self.init_speed = 0.0 # speed at initial position

        self.action_history = [] # history of action commands in one episode
        self.trajectory = [] # trajectory in one episode


    def get_odom(self):


    def compute_k(self):
        self.k = np.max(self.a)/self.max_speed
    
    def compute_actions(self):
        self.actions = [(acc,ang_v) for acc in self.a for ang_v in self.w]

    def compute_actions_dimension(self):
        return len(self.actions)

    def compute_dist_reward_scale(self):
        # scale the distance reward
        return 1 / (self.max_speed * self.N * self.dt)
    
    def compute_penalty_matrix(self):
        # scale the penalty value to [-1,0]
        scale_a = 1 / (np.max(self.a)*np.max(self.a))
        scale_w = 1 / (np.max(self.w)*np.max(self.w))
        p = -0.5 * np.matrix([[scale_a,0.0],[0.0,scale_w]])
        return p

    def compute_action_energy_cost(self,action):
        # scale the a and w to [0,1]
        a,w = self.actions[action]
        a /= np.max(self.a)
        w /= np.max(self.w)
        return np.abs(a) + np.abs(w)
    
    def dist_to_goal(self):
        return np.linalg.norm(self.goal - np.array([self.x,self.y]))

    def check_reach_goal(self):
        if self.dist_to_goal() <= self.goal_dis:
            self.reach_goal = True

    def reset_goal(self, goal_this):
        self.goal = np.array(goal_this)
        self.reach_goal = False

    def reset_state(self,current_velocity=np.zeros(2)):
        # only called when resetting the environment
        self.action_history.clear()
        self.trajectory.clear()
        self.x = self.start[0]
        self.y = self.start[1]
        self.theta = self.init_theta 
        self.speed = self.init_speed
        self.update_velocity(current_velocity)
        self.trajectory.append([self.x,self.y,self.theta,self.speed,self.velocity[0],self.velocity[1]]) 

    def get_robot_transform(self):
        # compute transformation from world frame to robot frame
        R_wr = np.matrix([[np.cos(self.theta),-np.sin(self.theta)],[np.sin(self.theta),np.cos(self.theta)]])
        t_wr = np.matrix([[self.x],[self.y]])
        return R_wr, t_wr

    def get_steer_velocity(self, speed=None, theta=None):
        if speed is None and theta is None:
            # Case when no parameters are provided
            return self.speed * np.array([np.cos(self.theta), np.sin(self.theta)])
        else:
            # Case when speed and theta are provided
            return speed * np.array([np.cos(theta), np.sin(theta)])

    def update_velocity(self,current_velocity=np.zeros(2),velocity=None,theta=None):
        if velocity is None and theta is None:
            steer_velocity = self.get_steer_velocity()
            self.velocity = steer_velocity + current_velocity
        else:
            steer_velocity = self.get_steer_velocity(velocity,theta)
            return steer_velocity + current_velocity

    def update_state(self,action,current_velocity):
        # update robot position in one time step
        self.update_velocity(current_velocity)
        dis = self.velocity * self.dt
        self.x += dis[0]
        self.y += dis[1]
        
        # update robot speed in one time step
        a,w = self.actions[action]
        
        # assume that water resistance force is proportion to the speed
        # self.speed += (a-self.k*self.speed) * self.dt
        # self.speed = np.clip(self.speed,0.0,self.max_speed)
        
        # update robot heading angle in one time step
        self.theta += w * self.dt

        # warp theta to [0,2*pi)
        while self.theta < 0.0:
            self.theta += 2 * np.pi
        while self.theta >= 2 * np.pi:
            self.theta -= 2 * np.pi
        # print(self.x, self.y, self.theta/np.pi * 180, self.velocity)

    def check_collision(self,obj_x,obj_y,obj_r):
        d = np.sqrt((self.x-obj_x)**2+(self.y-obj_y)**2)
        if d <= obj_r + self.r:
            return True
        else:
            return False

    # def compute_intersection(self,angle,obj_x,obj_y,obj_r):
    #     # compute the intersection point of a LiDAR beam with an object
    #     if np.abs(angle - np.pi/2) < 1e-03 or \
    #         np.abs(angle - 3*np.pi/2) < 1e-03:
    #         # vertical line
    #         M = obj_r*obj_r - (self.x-obj_x)*(self.x-obj_x)
            
    #         if M < 0.0:
    #             # no real solution
    #             return None
            
    #         x1 = self.x
    #         x2 = self.x
    #         y1 = obj_y - np.sqrt(M)
    #         y2 = obj_y + np.sqrt(M)
    #     else:
    #         K = np.tan(angle)
    #         a = 1 + K*K
    #         b = 2*K*(self.y-K*self.x-obj_y)-2*obj_x
    #         c = obj_x*obj_x + \
    #             (self.y-K*self.x-obj_y)*(self.y-K*self.x-obj_y) - \
    #             obj_r*obj_r
    #         delta = b*b - 4*a*c
            
    #         if delta < 0.0:
    #             # no real solution
    #             return None
            
    #         x1 = (-b-np.sqrt(delta))/(2*a)
    #         x2 = (-b+np.sqrt(delta))/(2*a)
    #         y1 = self.y+K*(x1-self.x)
    #         y2 = self.y+K*(x2-self.x)

    #     v1 = np.array([x1-self.x,y1-self.y])
    #     v2 = np.array([x2-self.x,y2-self.y])

    #     v = v1 if np.linalg.norm(v1) < np.linalg.norm(v2) else v2
    #     if np.linalg.norm(v) > self.perception.range:
    #         # beyond detection range
    #         return None
    #     if np.dot(v,np.array([np.cos(angle),np.sin(angle)])) < 0.0:
    #         # the intersection point is in the opposite direction of the beam
    #         return None
        
    #     return v

    def check_detection(self,obj_x,obj_y,obj_r):
        proj_pos = self.project_to_robot_frame(np.array([obj_x,obj_y]),False)
        
        if np.linalg.norm(proj_pos) > self.perception.range + obj_r:
            return False
        
        angle = np.arctan2(proj_pos[1],proj_pos[0])
        if angle < -0.5*self.perception.angle or angle > 0.5*self.perception.angle:
            return False
        
        return True
    
    def project_to_robot_frame(self,x,is_vector=True):
        assert isinstance(x,np.ndarray), "the input needs to be an numpy array"
        assert np.shape(x) == (2,)

        x_r = np.reshape(x,(2,1))

        R_wr, t_wr = self.get_robot_transform()

        R_rw = np.transpose(R_wr)
        t_rw = -R_rw * t_wr 

        if is_vector:
            x_r = R_rw * x_r
        else:
            x_r = R_rw * x_r + t_rw

        x_r.resize((2,))
        return np.array(x_r)            


    def perception_output(self,obstacles,robots):
        # TODO: remove LiDAR reflection computations and check dynamic obstacle observation error
        if self.reach_goal:
            return None, False, True
        
        self.perception.observation["static"].clear()

        ##### self observation (velocity and goal in self frame) #####
        # vehicle velocity wrt seafloor in self frame
        abs_velocity_r = self.project_to_robot_frame(self.velocity)

        # goal position in self frame
        goal_r = self.project_to_robot_frame(self.goal,False)

        self.perception.observation["self"] = list(np.concatenate((goal_r,abs_velocity_r)))


        ##### observation of other objects #####
        self.perception.observed_obs.clear()
        if self.cooperative:
            self.perception.observed_objs.clear()

        collision = False
        self.check_reach_goal()

        # static obstacles observation 
        for i,obs in enumerate(obstacles):
            if not self.check_detection(obs.x,obs.y,obs.r):
                continue

            self.perception.observed_obs.append(i)

            if not collision:
                collision = self.check_collision(obs.x,obs.y,obs.r)

            pos_r = self.project_to_robot_frame(np.array([obs.x,obs.y]),False)
            self.perception.observation["static"].append([pos_r[0],pos_r[1],obs.r])

        if self.cooperative:
            # dynamic objects observation
            for j,robot in enumerate(robots):
                if robot is self:
                    continue
                if robot.reach_goal:
                    # This robot is in the deactivate state, and abscent from the current map
                    continue
                if not self.check_detection(robot.x,robot.y,robot.detect_r):
                    continue

                self.perception.observed_objs.append(j)
                
                if not collision:
                    collision = self.check_collision(robot.x,robot.y,robot.r)

                pos_r = self.project_to_robot_frame(np.array([robot.x,robot.y]),False)
                v_r = self.project_to_robot_frame(robot.velocity)
                new_obs = list(np.concatenate((pos_r,v_r)))
                if j in self.perception.observation["dynamic"].copy().keys():
                    self.perception.observation["dynamic"][j].append(new_obs)
                    while len(self.perception.observation["dynamic"][j]) > self.perception.len_obs_history:
                        del self.perception.observation["dynamic"][j][0]
                else:
                    self.perception.observation["dynamic"][j] = [new_obs]

        # for rel_a in self.perception.beam_angles:
        #     angle = self.theta + rel_a

        #     # initialize the beam reflection as null
        #     # if a beam does not have reflection, set the reflection point distance
        #     # twice as long as the beam range, and set indicator to 0
        #     if np.abs(angle - np.pi/2) < 1e-03 or \
        #         np.abs(angle - 3*np.pi/2) < 1e-03:
        #         d = 2.0 if np.abs(angle - np.pi/2) < 1e-03 else -2.0
        #         x = self.x
        #         y = self.y + d*self.perception.range
        #     else:
        #         d = 2.0
        #         x = self.x + d * self.perception.range * np.cos(angle)
        #         y = self.y + d * self.perception.range * np.sin(angle)

        #     self.perception.reflections.append([x,y,0,-1])
            
        #     # compute the beam reflection from static obstcales 
        #     reflection_dist = np.infty  
        #     for i,obs in enumerate(obstacles):
                
        #         p = self.compute_intersection(angle,obs.x,obs.y,obs.r)

        #         if p is None:
        #             continue

        #         if self.perception.reflections[-1][-1] != 0:
        #             # check if the current intersection point is closer
        #             if np.linalg.norm(p) >= reflection_dist: 
        #                 continue
                
        #         reflection_dist = np.linalg.norm(p)
        #         self.perception.reflections[-1] = [p[0]+self.x,p[1]+self.y,1,i]

        #     # compute the beam reflection from dynamic objects
        #     for j,robot in enumerate(robots):
        #         if robot is self:
        #             continue
        #         if robot.reach_goal:
        #             # This robot is in the deactivate state, and abscent from the current map
        #             continue

        #         p = self.compute_intersection(angle,robot.x,robot.y,robot.detect_r)

        #         if p is None:
        #             continue

        #         if self.perception.reflections[-1][-1] != 0:
        #             # check if the current intersection point is closer
        #             if np.linalg.norm(p) >= reflection_dist: 
        #                 continue
            
        #         reflection_dist = np.linalg.norm(p)
        #         self.perception.reflections[-1] = [p[0]+self.x,p[1]+self.y,2,j]

        #     tp = self.perception.reflections[-1][2]
        #     idx = self.perception.reflections[-1][-1]

        #     if tp == 1:
        #         if idx not in self.perception.observed_obs:
        #             # add the new static object observation into the observation
        #             self.perception.observed_obs.append(idx)
        #             obs = obstacles[idx]

        #             if not collision:
        #                 collision = self.check_collision(obs.x,obs.y,obs.r)

        #             pos_r = self.project_to_robot_frame(np.array([obs.x,obs.y]),False)
        #             self.perception.observation["static"].append([pos_r[0],pos_r[1],obs.r])
                    
        #     elif tp == 2 and self.cooperative:
        #         if idx not in self.perception.observed_objs:
        #             # In a cooperative agent, also include dynamic object observation
        #             self.perception.observed_objs.append(idx)
        #             robot = robots[idx]
        #             assert robot.reach_goal is False, "robots that reach goal must be removed!"
                    
        #             if not collision:
        #                 collision = self.check_collision(robot.x,robot.y,obs.r)

        #             pos_r = self.project_to_robot_frame(np.array([robot.x,robot.y]),False)
        #             v_r = self.project_to_robot_frame(robot.velocity)
        #             new_obs = list(np.concatenate((pos_r,v_r)))
        #             if idx in self.perception.observation["dynamic"].copy().keys():
        #                 self.perception.observation["dynamic"][idx].append(new_obs)
        #                 while len(self.perception.observation["dynamic"][idx]) > self.perception.len_obs_history:
        #                     del self.perception.observation["dynamic"][idx][0]
        #             else:
        #                 self.perception.observation["dynamic"][idx] = [new_obs]

        if self.cooperative:
            for idx in self.perception.observation["dynamic"].copy().keys():
                if idx not in self.perception.observed_objs:
                    # remove the observation history if the object is not observed in the current step
                    del self.perception.observation["dynamic"][idx]

        self_state = copy.deepcopy(self.perception.observation["self"])
        static_states = copy.deepcopy(self.perception.observation["static"])
        if self.cooperative:
            # remove object indices 
            dynamic_states = copy.deepcopy(list(self.perception.observation["dynamic"].values()))
            idx_array = []
            for idx,obs_history in enumerate(dynamic_states):
                # pad the dynamic observation and save the indices of exact lastest element
                idx_array.append([idx,len(obs_history)-1])
                while len(obs_history) < self.perception.len_obs_history:
                    obs_history.append([0.,0.,0.,0.])
            obs = (self_state,static_states,dynamic_states,idx_array)
            return obs, collision, self.reach_goal
        else:
            return (self_state,static_states), collision, self.reach_goal



            




            

            

                     



