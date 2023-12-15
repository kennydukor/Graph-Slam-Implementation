import numpy as np

class GraphSLAM2D():
    def __init__(self, robot, world_size, num_landmarks, time_step) -> None:
        '''
        robot: robot object
        world_size: size of world (square)
        num_landmarks: number of landmarks
        time_step (N): time step
        omega: constraint matrix
        xi: constraint vector
        '''
        self.robot = robot
        self.world_size = world_size
        self.num_landmarks = num_landmarks
        self.time_step = time_step
        self.omega = None
        self.xi = None

    def initialize_constraints(self): 
        '''
        Initialize constraint matrix and vector
        '''
        dim = 2 * (self.time_step + self.num_landmarks)  # 2D pose for robot, 2D pose for each landmark

        self.omega = np.zeros((dim, dim)) # initial matrix of all zeros (row and column size)

        self.xi = np.zeros((dim, 1)) # initial vector of all zeros

        # Set initial location to the center of the world
        self.omega[0, 0] = 1
        self.omega[1, 1] = 1
        self.xi[0] = self.world_size[0] / 2
        self.xi[1] = self.world_size[1] / 2

        return self.omega, self.xi
    
    def slam(self, motion_noise, measurement_noise, data):
        '''
        poses: list of poses
        world_size: size of world (square)
        num_landmarks: number of landmarks
        landmarks: list of landmarks
        motion_noise: noise associated with motion
        measurement_noise: noise associated with measurement
        '''

        # initialize constraint matrix/vector
        self.omega, self.xi = self.initialize_constraints()

        num_poses = len(data) # number of poses
        
        # update matrix/vector for each measurement
        for i in range(num_poses):
            measurement, pose = data[i]

            pose_index = 2 * i # corrected pose index

            # process measurements
            for m in measurement:
                landmark_index, dx, dy = m

                landmark_index = 2 * (self.time_step + landmark_index) # corrected landmark index

                # Update omega and xi for measurements
                self.omega[pose_index:pose_index+2, pose_index:pose_index+2] += np.identity(2) * 1.0 / measurement_noise
                self.omega[landmark_index:landmark_index+2, landmark_index:landmark_index+2] += np.identity(2) * 1.0 / measurement_noise
                
                self.omega[pose_index:pose_index+2, landmark_index:landmark_index+2] += np.identity(2) * -1.0 / measurement_noise
                self.omega[landmark_index:landmark_index+2, pose_index:pose_index+2] += np.identity(2) * -1.0 / measurement_noise

                # Update xi for measurements
                self.xi[pose_index:pose_index+2] += np.array([[dx], [dy]]) * -1.0 / measurement_noise
                self.xi[landmark_index:landmark_index+2] += np.array([[dx], [dy]]) * 1.0 / measurement_noise
                # self.xi[landmark_index:landmark_index+2] += np.array([[pose[0]], [pose[1]]]) * 1.0 / measurement_noise
                
                # Process motion (if there's a next pose)
                if i < num_poses - 1:
                    next_pose = data[i + 1][1]
                    dx, dy = next_pose[0] - pose[0], next_pose[1] - pose[1]
                    
                    # Update omega for motion
                    self.omega[pose_index:pose_index+2, pose_index:pose_index+2] += np.identity(2) / motion_noise
                    self.omega[pose_index+2:pose_index+4, pose_index+2:pose_index+4] += np.identity(2) / motion_noise
                    self.omega[pose_index:pose_index+2, pose_index+2:pose_index+4] -= np.identity(2) / motion_noise
                    self.omega[pose_index+2:pose_index+4, pose_index:pose_index+2] -= np.identity(2) / motion_noise

                    # Update xi for motion
                    self.xi[pose_index:pose_index+2] -= np.array([[dx], [dy]]) / motion_noise
                    self.xi[pose_index+2:pose_index+4] += np.array([[dx], [dy]]) / motion_noise
        
        # Regularization: Add a small value to the diagonal elements
        np.fill_diagonal(self.omega, self.omega.diagonal() + 1e-10)

        # Calculate mu using pseudo-inverse as a fallback
        try:
            mu = np.dot(np.linalg.inv(self.omega), self.xi)
        except np.linalg.LinAlgError:
            mu = np.dot(np.linalg.pinv(self.omega), self.xi)
        
        return mu
    
# class GraphSLAM3D():
#     def __init__(self, robot, world_size, num_landmarks, time_step) -> None:
#         '''
#         robot: robot object
#         world_size: size of world
#         num_landmarks: number of landmarks
#         time_step (N): number of time steps
#         omega: constraint matrix
#         xi: constraint vector
#         '''
#         self.robot = robot
#         self.world_size = world_size
#         self.num_landmarks = num_landmarks
#         self.time_step = time_step
#         self.omega = None
#         self.xi = None

#     def initialize_constraints(self):
#         '''
#         Initialize constraint matrix and vector
#         '''
#         # Each pose has 3 components (x, y, theta) and each landmark has 2 (x, y)
#         dim = 3 * self.time_step + 2 * self.num_landmarks

#         self.omega = np.zeros((dim, dim))
#         self.xi = np.zeros((dim, 1))

#         # Set initial location and orientation
#         self.omega[0:3, 0:3] = np.eye(3)
#         self.xi[0:2, 0] = [0, 0] #[self.world_size[0] / 2, self.world_size[1] / 2]

#         return self.omega, self.xi

#     def slam(self, motion_noise, measurement_noise, data):
#         '''
#         motion_noise: noise associated with motion
#         measurement_noise: noise associated with measurement
#         data: list of tuples containing measurements and motion
#         '''
#         self.omega, self.xi = self.initialize_constraints()

#         for i in range(len(data)):
#             measurement, motion = data[i]

#             # Update for motion
#             pose_index = 3 * i
#             next_pose_index = pose_index + 3
#             dx, dy, dtheta = motion

#             # Update omega for motion
#             motion_update = np.eye(3) / motion_noise
#             self.omega[pose_index:pose_index+3, pose_index:pose_index+3] += motion_update
#             if i < len(data) - 1:
#                 self.omega[next_pose_index:next_pose_index+3, next_pose_index:next_pose_index+3] += motion_update
#                 self.omega[pose_index:pose_index+3, next_pose_index:next_pose_index+3] -= motion_update
#                 self.omega[next_pose_index:next_pose_index+3, pose_index:pose_index+3] -= motion_update

#                 # Update xi for motion
#                 self.xi[pose_index:pose_index+3] -= np.array([[dx], [dy], [dtheta]]) / motion_noise
#                 self.xi[next_pose_index:next_pose_index+3] += np.array([[dx], [dy], [dtheta]]) / motion_noise

#             # Update for measurements
#             for m in measurement:
#                 landmark_index, dx, dy = m
#                 landmark_matrix_index = 3 * self.time_step + 2 * landmark_index

#                 # Update omega for measurements
#                 measurement_update = np.eye(2) / measurement_noise
#                 self.omega[pose_index:pose_index+2, pose_index:pose_index+2] += measurement_update
#                 self.omega[landmark_matrix_index:landmark_matrix_index+2, landmark_matrix_index:landmark_matrix_index+2] += measurement_update
#                 self.omega[pose_index:pose_index+2, landmark_matrix_index:landmark_matrix_index+2] -= measurement_update
#                 self.omega[landmark_matrix_index:landmark_matrix_index+2, pose_index:pose_index+2] -= measurement_update

#                 # Update xi for measurements
#                 self.xi[pose_index:pose_index+2] -= np.array([[dx], [dy]]) / measurement_noise
#                 self.xi[landmark_matrix_index:landmark_matrix_index+2] += np.array([[dx], [dy]]) / measurement_noise

#         # Solve for best estimate
#         mu = np.dot(np.linalg.inv(self.omega), self.xi)
#         return mu