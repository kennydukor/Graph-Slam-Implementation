import random
import numpy as np
from math import *

class Robot:
    def __init__(self, world, sensor_range, sensor_noise=2.0, motion_noise=2.0, turning_noise=2.0, x=0.0, y=0.0, theta=15):
        '''
        x: x position
        y: y position
        theta: orientation (yaw) in degrees
        world: world object
        sensor_range: maximum distance robot can sense
        sensor_noise: measurement noise
        motion_noise: motion noise
        turning_noise: turning noise
        '''
        self.x = x
        self.y = y
        self.world = world
        self.sensor_range = sensor_range
        self.sensor_noise = sensor_noise # measurement noise
        self.motion_noise = motion_noise
        self.turning_noise = turning_noise
        if theta is None:
            self.theta = random.random() * 2.0 * np.pi
        else:
            self.theta = np.deg2rad(theta)
        self.state_space = np.array([[self.x, self.y, self.theta]]).T

    def measurement_prob(self, measurements):
        '''
        measurements: list of landmark indices that are within sensor range (measurement range) of the robot
        '''
        # calculate the correct measurement
        correct_measurements = self.sense()

        # calculate the probability of the measurements
        prob = 1.0
        for i in range(len(measurements)):
            measurement = measurements[i]
            correct_measurement = correct_measurements[i]

            # calculate distance between robot and landmark
            distance = np.sqrt((self.x - self.landmarks[correct_measurement].x)**2 + (self.y - self.landmarks[correct_measurement].y)**2)

            # calculate Gaussian probability
            prob *= self.gaussian(distance, self.sensor_noise, measurement)

        return prob
    
    def gaussian(self, mu, sigma, x):
        '''
        mu: mean
        sigma: standard deviation
        x: measurement
        '''
        # calculate probability
        prob = np.exp(-((mu - x)**2) / (sigma**2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma**2))

        return prob

    def control_input(self, v=1.0, w=np.deg2rad(15.0)):
        """
        Calculate control input.
        """
        self.v = v  # linear velocity [m/s]
        self.w = w  # angular velocity [rad/s]
        return np.array([[self.v, self.w]]).T

    def observe_landmarks(self, true_pose, pose_DR, control_input):
        """
        Simulate the observation of landmarks using the robot's control input.
        control_input: control input vector [linear_velocity, angular_velocity]
        """
        # Define noise parameters for range and bearing measurements
        range_bearing_noise = np.diag([0.01, np.deg2rad(0.010)])**2

        # Define noise parameters for control inputs [linear_velocity, angular_velocity]
        control_noise = np.diag([0.1, np.deg2rad(1.0)])**2

        # Add noise to control inputs
        noisy_linear_velocity = control_input[0, 0] + np.random.randn() * control_noise[0, 0]
        noisy_angular_velocity = control_input[1, 0] + np.random.randn() * control_noise[1, 1]
        noisy_control = np.array([[noisy_linear_velocity, noisy_angular_velocity]]).T

        # Predict the true state with input
        true_state = self.motion_model(true_pose, control_input)

        # Predict the dead reckoning state with noisy input
        dead_reckoning_state = self.motion_model(pose_DR, noisy_control)

        # Initialize an empty array for storing landmark observations
        landmark_observations = np.zeros((0, 3))

        for i, landmark in enumerate(self.world.landmarks):
            dx = landmark[0] - true_state[0]
            dy = landmark[1] - true_state[1]
            distance = np.sqrt(dx**2 + dy**2)[0]
       
            bearing = self.normalize_angle(np.arctan2(dy, dx) - true_state[2, 0])[0]

            if distance <= self.sensor_range:
                distance_with_noise = distance + np.random.randn() * range_bearing_noise[0, 0]
                bearing_with_noise = bearing + np.random.randn() * range_bearing_noise[1, 1]
                observation = np.array([[distance_with_noise, bearing_with_noise, i]])
                landmark_observations = np.vstack((landmark_observations, observation))

        return true_state, landmark_observations, dead_reckoning_state, noisy_control

    def motion_model(self, x, u, dt=2):
        """
        Motion model for a differential drive robot.

        :param x: Current state of the robot [x, y, yaw]
        :param u: Control inputs [v, w]
        :return: New state of the robot
        """
        F = np.array([[1.0, 0, 0],
                    [0, 1.0, 0],
                    [0, 0, 1.0]])

        B = np.array([[dt * np.cos(x[2, 0]), 0],
                    [dt * np.sin(x[2, 0]), 0],
                    [0.0, dt]])

        x = F @ x + B @ u
        x[2, 0] = self.normalize_angle(x[2, 0])  # Normalize yaw
        return x

    def normalize_angle(self, angle):
        """
        Normalize angle to be within [-pi, pi].

        :param angle: Angle in radians
        :return: Normalized angle
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle
    
    