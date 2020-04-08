import numpy as np
import math

class DiscreteEnvironment:
    def __init__(self, size=32):
        self.size = size
        self.robot_hl = 2

    def generateTrajectory(self, start=None, goal=None):
        if start is None:
            start = np.random.randint(self.robot_hl, self.size-self.robot_hl, size=(2), dtype=int)
            goal = np.random.randint(self.robot_hl, self.size-self.robot_hl, size=(2), dtype=int)

        diff = goal - start
        if np.abs(diff[0]) < np.abs(diff[1]):
            # Move along y direction first
            action = np.array([0, np.sign(diff[1])])
            steps = np.abs(diff[1]) - np.abs(diff[0])
        elif np.abs(diff[0]) > np.abs(diff[1]):
            # move along x direction first
            action = np.array([np.sign(diff[0]), 0])
            steps = np.abs(diff[0]) - np.abs(diff[1])
        else:
            # move along diagonal
            action = np.sign(diff)
            steps = np.abs(diff[0])

        traj1, s = self.__uniform_traj(steps, action, start)

        # finish the remaining diagonal
        diff = goal - s
        action = np.sign(diff)
        steps = np.abs(diff[0])

        traj2, s = self.__uniform_traj(steps, action, s)
        traj = np.concatenate((traj1, traj2), axis=0)
        return start, goal, traj

    def generateImage(self, start, goal):
        result = np.zeros((self.size, self.size))
        self.__uniform_fill(result, goal, 127)
        self.__uniform_fill(result, start, 255) #start overwrites goal
        return result

    def __uniform_fill(self, imgArray, refPos, refVal):
        for xdiff in range(-self.robot_hl, self.robot_hl+1):
            for ydiff in range(-self.robot_hl, self.robot_hl+1):
                imgArray[refPos[0] + xdiff][refPos[1] + ydiff] = refVal


    def __uniform_traj(self, steps, action, start):
        traj = np.zeros((steps, 6), dtype=int)
        s = start
        for i in range(steps):
            sn = s + action
            traj[i] = np.concatenate((s, action, sn), axis=0)
            s = sn
        return traj, s
