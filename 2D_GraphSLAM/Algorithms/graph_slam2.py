import numpy as np
import itertools
import math, copy
import matplotlib.pyplot as plt

class Edge:
    def __init__(self):
        # Initialize the error vector
        self.e = np.zeros((3, 1))

        # Initialize other attributes
        self.distance_node1 = 0
        self.distance_node2 = 0
        self.yaw1 = 0
        self.yaw2 = 0
        self.angle1 = 0
        self.angle2 = 0
        self.node1_id = 0
        self.node2_id = 0
        self.optimized_pose = None


class GraphSLAM:
    def __init__(self, pose_list, observation_list, landmarks, graphics_radius=0.3):
        self.pose_list = pose_list
        self.observation_list = observation_list
        self.landmarks = landmarks
        self.graphics_radius = graphics_radius
        self.edges = []

        self.generate_edges()
        # self.visualize_measurements()

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

    def generate_edges(self):
        """
        Generate edges from node observations.
        """
        print(self.observation_list)
        zids = list(itertools.combinations(range(len(self.observation_list)), 2))
        # print("Node combinations: \n", zids)

        for i in range(self.pose_list.shape[1]):
            # Check if observation_list[i] is not empty and has at least one row and three columns
            if len(self.observation_list[i]) > 0 and self.observation_list[i].shape[1] >= 3:
                print("Node {} observed landmark with ID {}".format(i, self.observation_list[i][0, 2]))
            else:
                print("Node {} has no observations or insufficient data".format(i))

        for (t1, t2) in zids:
            if len(self.observation_list[t1]) > 0 and len(self.observation_list[t2]) > 0:
                self.process_edge(t1, t2)

    def process_edge(self, time1, time2):
        """
        Process and create an edge from two nodes.
        """

        pose_x1, pose_y1, pose_yaw1 = self.extract_node_state(time1)
        pose_x2, pose_y2, pose_yaw2 = self.extract_node_state(time2)

        observation_distance1, observation_angle1, _ = self.observation_list[time1][0]
        observation_distance2, observation_angle2, _ = self.observation_list[time2][0]

        self.transformed_angle1 = self.normalize_angle(pose_yaw1 + observation_angle1)
        self.transformed_angle2 = self.normalize_angle(pose_yaw2 + observation_angle2)

        projected_x1, projected_x2, projected_y1, projected_y2 = self.project_observations(
            observation_distance1, observation_distance2, 
            self.transformed_angle1, self.transformed_angle2)

        new_edge = self.create_edge(time1, time2, pose_x1, pose_y1, pose_x2, pose_y2, 
                                    pose_yaw1, pose_yaw2, 
                                    observation_distance1, observation_distance2, 
                                    observation_angle1, observation_angle2, 
                                    projected_x1, projected_x2, projected_y1, projected_y2)
        self.edges.append(new_edge)

    def extract_node_state(self, t):
        """
        Extracts the state (position and yaw) of a node.
        """
        return self.pose_list[0, t], self.pose_list[1, t], self.pose_list[2, t]

    def project_observations(self, distance_node1, distance_node2, tangle1, tangle2):
        """
        Project observations onto the XY plane.
        """
        tmp1 = distance_node1 * math.cos(tangle1)
        tmp2 = distance_node2 * math.cos(tangle2)
        tmp3 = distance_node1 * math.sin(tangle1)
        tmp4 = distance_node2 * math.sin(tangle2)
        return tmp1, tmp2, tmp3, tmp4

    def create_edge(self, t1, t2, x1, y1, x2, y2, yaw1, yaw2, distance_node1, distance_node2, angle1, angle2, tmp1, tmp2, tmp3, tmp4):
        """
        Create an edge from the provided information.
        """
        edge = Edge()
        edge.error_vector = np.array([[x2 - x1 - tmp1 + tmp2],
                           [y2 - y1 - tmp3 + tmp4],
                           [self.normalize_angle(yaw2 - yaw1 - self.transformed_angle1 + self.transformed_angle2)]])

        edge.distance_node1, edge.distance_node2 = distance_node1, distance_node2
        edge.yaw1, edge.yaw2 = yaw1, yaw2
        edge.angle1, edge.angle2 = angle1, angle2
        edge.node1_id, edge.node2_id = t1, t2

        print("For nodes", (t1, t2))
        print("Added edge with errors: \n", edge.error_vector)
        return edge

    def visualize_measurements(self):
        """
        Visualize the measurements and nodes.
        """
        for edge in self.edges:
            t1, t2 = edge.node1_id, edge.node2_id
            x1, y1, _ = self.extract_node_state(t1)
            x2, y2, _ = self.extract_node_state(t2)

            # plt.plot(self.landmarks[0, 0], self.landmarks[0, 1], "*k", markersize=20)
            plt.plot([x1, x2], [y1, y2], '.', markersize=50, alpha=0.8)
            self.draw_observations(x1, y1, edge.yaw1, edge.distance_node1, edge.angle1)
            self.draw_observations(x2, y2, edge.yaw2, edge.distance_node2, edge.angle2)
        for landmark in self.landmarks:
            plt.plot(landmark[0], landmark[1], "*k", markersize=20)

        plt.legend()
        plt.grid()
        plt.show()

    def draw_observations(self, x, y, yaw, d, angle):
        """
        Draw observation lines and points.
        """
        tangle = self.normalize_angle(yaw + angle)
        tmp_x = d * math.cos(tangle)
        tmp_y = d * math.sin(tangle)

        plt.plot([x, x + self.graphics_radius * np.cos(yaw)],
                 [y, y + self.graphics_radius * np.sin(yaw)], 'r')
        plt.plot([x, x + tmp_x], [y, y], label="obs x")
        plt.plot([x, x], [y, y + tmp_y], label="obs y")
        plt.plot(x + tmp_x, y + tmp_y, 'o')

    # Perfom graphSLAM
    def perform_slam(self, pose_DR_store, true_pose_store, STATE_SIZE, C_SIGMA1, C_SIGMA2, C_SIGMA3, number_of_iterations=10):
        """
        Perform GraphSLAM calculations over multiple iterations.

        :param pose_DR_store: Initial dead reckoning trajectory
        :param true_pose_store: Ground truth trajectory
        :param STATE_SIZE: Size of the state vector
        :param C_SIGMA1, C_SIGMA2, C_SIGMA3: Covariance values
        :param number_of_iterations: Number of iterations for SLAM calculations
        """
        
        epsilon = 1e-10  # Small value to prevent singular matrix
        n = len(self.pose_list[0]) * STATE_SIZE  # Total size of the system

        # Create a deep copy of hxDR for self.optimized_pose (clean datastructure)
        self.optimized_pose = copy.deepcopy(pose_DR_store)

        for iteration in range(number_of_iterations):
            H, b = self.initialize_matrices(n)

            for edge in self.edges:
                self.update_matrices(H, b, edge, STATE_SIZE, C_SIGMA1, C_SIGMA2, C_SIGMA3)

            # Apply origin constraint for visualization
            H[0:STATE_SIZE, 0:STATE_SIZE] += np.identity(STATE_SIZE)

            # Regularization to avoid singularity
            H_reg = H + np.eye(n) * epsilon

            # Check if the matrix is singular
            if np.linalg.matrix_rank(H_reg) < n:
                print("Warning: Regularized matrix H is still singular")
                continue

            try:
                # Solve the linear system
                dx = np.linalg.solve(H_reg, -b)
            except np.linalg.LinAlgError:
                print("Linear algebra error encountered. Skipping iteration.")
                continue

            # Update the state estimate
            for i in range(len(self.pose_list[0])):
                self.optimized_pose[0:3, i] += dx[i * 3:i * 3 + 3, 0]

            print(f"Iteration {iteration}: dx = {dx}")

        # After all iterations
        print("Final SLAM state: \n", self.optimized_pose)
        print("\ngraphSLAM localization error: ", np.sum((self.optimized_pose - true_pose_store) ** 2))
        print("Odom localization error: ", np.sum((pose_DR_store - true_pose_store) ** 2))

    def initialize_matrices(self, n):
        """
        Initialize the information matrix and vector.

        :param n: Total size of the system
        :param STATE_SIZE: Size of the state vector
        :return: Initialized matrices H and b
        """
        H = np.zeros((n, n))
        b = np.zeros((n, 1))
        return H, b

    def update_matrices(self, H, b, edge, STATE_SIZE, C_SIGMA1, C_SIGMA2, C_SIGMA3):
        """
        Update the information matrix and vector based on an edge.

        :param H: Information matrix
        :param b: Information vector
        :param edge: Edge to use for the update
        :param STATE_SIZE: Size of the state vector
        :param C_SIGMA1, C_SIGMA2, C_SIGMA3: Covariance values
        """
        node1_id = edge.node1_id * STATE_SIZE
        node2_id = edge.node2_id * STATE_SIZE

        t1 = edge.yaw1 + edge.angle1
        A = np.array([[-1.0, 0, edge.distance_node1 * math.sin(t1)],
                      [0, -1.0, -edge.distance_node1 * math.cos(t1)],
                      [0, 0, -1.0]])

        t2 = edge.yaw2 + edge.angle2
        B = np.array([[1.0, 0, -edge.distance_node2 * math.sin(t2)],
                      [0, 1.0, edge.distance_node2 * math.cos(t2)],
                      [0, 0, 1.0]])

        sigma = np.diag([C_SIGMA1, C_SIGMA2, C_SIGMA3])
        Rt1 = self.calc_rotational_matrix(t1)
        Rt2 = self.calc_rotational_matrix(t2)
        edge.omega = np.linalg.inv(Rt1 @ sigma @ Rt1.T + Rt2 @ sigma @ Rt2.T)

        H[node1_id:node1_id + STATE_SIZE, node1_id:node1_id + STATE_SIZE] += A.T @ edge.omega @ A
        H[node1_id:node1_id + STATE_SIZE, node2_id:node2_id + STATE_SIZE] += A.T @ edge.omega @ B
        H[node2_id:node2_id + STATE_SIZE, node1_id:node1_id + STATE_SIZE] += B.T @ edge.omega @ A
        H[node2_id:node2_id + STATE_SIZE, node2_id:node2_id + STATE_SIZE] += B.T @ edge.omega @ B

        b[node1_id:node1_id + STATE_SIZE] += (A.T @ edge.omega @ edge.error_vector)
        b[node2_id:node2_id + STATE_SIZE] += (B.T @ edge.omega @ edge.error_vector)

    def calc_rotational_matrix(self, theta):
        """
        Calculate a rotational matrix for a given angle theta.

        :param theta: Rotation angle in radians.
        :return: Rotational matrix.
        """
        # Assuming 2D rotation here. Modify as needed for your application.
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        return np.array([[cos_theta, -sin_theta, 0],
                         [sin_theta, cos_theta, 0],
                         [0, 0, 1]])


    def visualize_updated_pose(self, true_pose_store):
        """
        Visualize the updated poses from the SLAM calculation.

        :param optimized_pose: The updated positions and orientations after SLAM
        """
        for i in range(self.optimized_pose.shape[1]):
            x, y, yaw = self.optimized_pose[0, i], self.optimized_pose[1, i], self.optimized_pose[2, i]

            # Plot the updated position
            plt.plot(x, y, 'o', markersize=10, label=f"Node {i}")

            # Plot the orientation
            plt.plot([x, x + self.graphics_radius * np.cos(yaw)],
                     [y, y + self.graphics_radius * np.sin(yaw)], 'r')
            
        # Plot the true poses
        for i in range(true_pose_store.shape[1]):
            x, y, yaw = true_pose_store[0, i], true_pose_store[1, i], true_pose_store[2, i]

            # Plot the true position
            plt.plot(x, y, 'x', markersize=10, label=f"True Node {i}" if i == 0 else "")

            # Plot the true orientation
            plt.plot([x, x + self.graphics_radius * np.cos(yaw)],
                     [y, y + self.graphics_radius * np.sin(yaw)], 'g')

        # Optionally, plot the RFID locations as well
        for rfid in self.landmarks:
            plt.plot(rfid[0], rfid[1], "*k", markersize=20)

        plt.legend()
        plt.grid()
        plt.title("Updated Poses from SLAM")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.show()