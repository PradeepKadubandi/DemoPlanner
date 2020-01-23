# from shapely.geometry import box
# from shapely.geometry import Point
import numpy as np
import math
import matplotlib.pyplot as plt

class VisualEnvironment:
    def __init__(self, size=32.0, includeRobot=True, shouldInit=True, addObstacles=True, grad_bg=1.0, grad_robot=3.0, grad_obs=2.0):
        self.size = size
        self.includeRobot = includeRobot
        self.robot_hl = 2.0
        self.data = None
        self.addObstacles = addObstacles
        self.grad_bg = grad_bg
        self.grad_robot = grad_robot
        self.grad_obs = grad_obs
        if shouldInit:
            self.initialize()

    def safe_robot_center(self, any_center):
        return np.clip(any_center, self.robot_hl, self.size-self.robot_hl)

    def initialize(self):
        self.num_obstacles = 0
        if self.addObstacles:
            self.num_obstacles = np.random.randint(low=2, high=7)
        self.type_obstacle = np.random.randint(0, 2, size=self.num_obstacles)
        self.center_obstacles = np.floor(self.size * np.random.rand(self.num_obstacles, 2))
        # spec: Radius for circle and half_length for square
        # Min value is 2 and maximum 8
        self.spec_obstacle = 2 + 6 * np.random.rand(self.num_obstacles)
        # Clip the obstacle size not to exceed boundaries
        upper_bounds = self.size - self.center_obstacles
        self.spec_obstacle = np.min(np.stack((self.spec_obstacle,
                                            self.center_obstacles[:,0],
                                            self.center_obstacles[:,1],
                                            upper_bounds[:,0],
                                            upper_bounds[:,1]),axis=1), axis=1)
        self.robot_center = self.safe_robot_center(np.floor(self.size * np.random.rand(2)))
        
    def with_new_robot_position(self, new_robot_center):
        '''
        Returns a new object with all other things being shared except robot_position
        '''
        result = VisualEnvironment(self.size, includeRobot=True, shouldInit=False)
        result.num_obstacles = self.num_obstacles
        # Intentionally sharing the obstacle arrays as these should be common across copies
        result.type_obstacle = self.type_obstacle
        result.center_obstacles = self.center_obstacles
        result.spec_obstacle = self.spec_obstacle
        result.grad_bg = self.grad_bg
        result.grad_obs = self.grad_obs
        result.grad_robot = self.grad_robot
        result.robot_center = new_robot_center
        return result
        
    def build_data(self):
        '''
        Builds a numpy array with different integer range values for empty space, obstacles, robot
        '''
        self.data = np.array([[self.value_for_position([x, y]) for y in range(int(self.size))] for x in range(int(self.size))])

    def generate_trajectory(self, length):
        envs = []
        controls = []
        stepsize = 3
        delta_t = 1
        direction = np.random.choice([-1, +1], size=(2))
        current_env = self
        for i in range(length):
            control = np.floor(np.random.uniform(0.0, stepsize, size=(2)) * direction)
            new_robot_pos = self.safe_robot_center(np.floor(current_env.robot_center + delta_t * control))
            current_env = current_env.with_new_robot_position(new_robot_pos)
            envs.append(current_env)
            controls.append(control)
        return envs, controls, direction

    def value_for_position(self, pos):
        world_center = np.array([self.size/2, self.size/2])
        world_center_value = 0.0
        value_from_world = world_center_value - self.square_obs_distance(world_center, pos) * self.grad_bg

        obs_center_value = 70.0
        min_distance_from_any_obs_center = None
        value_from_obstacle = 0.0
        for i in range(self.num_obstacles):
            if self.type_obstacle[i] == 0:
                dist = self.square_obs_distance(self.center_obstacles[i, :], pos)
            else:
                dist = self.circular_obs_distance(self.center_obstacles[i, :], pos)
            if dist < self.spec_obstacle[i]:
                if min_distance_from_any_obs_center is None:
                    min_distance_from_any_obs_center = dist
                else:
                    min_distance_from_any_obs_center = min(min_distance_from_any_obs_center, dist)
        if min_distance_from_any_obs_center is not None:
            value_from_obstacle = obs_center_value - self.grad_obs * min_distance_from_any_obs_center

        robot_center_value = 255.0
        value_from_robot = 0.0
        if self.includeRobot:
            dist = self.square_obs_distance(self.robot_center, pos)
            if dist <= self.robot_hl:
                value_from_robot = robot_center_value - self.grad_robot * dist

        return value_from_world + value_from_obstacle + value_from_robot
        
    def circular_obs_distance(self, center, point):
        return np.linalg.norm(center-point)
    
    def square_obs_distance(self, center, point):
        return np.max(np.abs(center-point))
    
    def plot_data(self):
        '''
        Plots the raw data generated after build_array
        '''
        if self.data is None:
            self.build_data()
        plt.imshow(self.data)