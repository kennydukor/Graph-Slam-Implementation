import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GraphSLAM:
    def __init__(self):
        self.poses = []
        self.landmarks = []
        self.edges = []
        self.loop_closures = []

    def read_dataset(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('VERTEX3'):
                    vertex_id, pose_values = parse_vertex3(line)
                    self.add_pose(vertex_id, pose_values)
                elif line.startswith('LANDMARK3'):
                    landmark_id, landmark_values = parse_landmark3(line)
                    self.add_landmark(landmark_id, landmark_values)
                elif line.startswith('EDGE3'):
                    id_out, id_in, measurement, info_matrix = parse_edge3(line)
                    self.add_edge(id_out, id_in, measurement, info_matrix)
                elif line.startswith('LOOP_CLOSURE'):
                    pose_index1, pose_index2, measurement, info_matrix = parse_loop_closure(line)
                    self.add_loop_closure(pose_index1, pose_index2, measurement, info_matrix)

    def get_poses_from_ids(self, id_out, id_in, poses_array):
        # Ensure that the indices are integers
        id_out = int(id_out)
        id_in = int(id_in)

        # If poses_array is a list, use the index directly
        if isinstance(poses_array, list):
            pose_out = poses_array[id_out][1]
            pose_in = poses_array[id_in][1]
        else:
            # If poses_array is a NumPy array, use NumPy-style indexing
            pose_out = poses_array[id_out, :]
            pose_in = poses_array[id_in, :]

        return pose_out, pose_in

    def add_pose(self, vertex_id, pose):
        self.poses.append((vertex_id, np.array(pose)))

    def add_landmark(self, landmark_id, landmark):
        self.landmarks.append((landmark_id, landmark))

    def add_edge(self, id_out, id_in, measurement, info_matrix):
        self.edges.append((id_out, id_in, measurement, info_matrix))

    def add_loop_closure(self, pose_index1, pose_index2, measurement, info_matrix):
        self.loop_closures.append((pose_index1, pose_index2, measurement, info_matrix))

    def construct_initial_guess(self):
        initial_guess = []

        for edge in self.edges:
            id_out, id_in, measurement, _ = edge
            pose_out, pose_in = self.get_poses_from_ids(id_out, id_in, self.poses)

            delta_pose = pose_in - pose_out
            expected_measurement = np.linalg.norm(delta_pose)

            # Ensure the shapes are compatible for broadcasting
            measurement = np.array(measurement)
            expected_measurement = np.array(expected_measurement)
            delta_pose = np.array(delta_pose)

            # Adjust the initial guess based on the measurement and expected value
            adjusted_pose = pose_out + (measurement / expected_measurement) * delta_pose

            # Ensure the sizes of pose_out and delta_pose match
            adjusted_pose[:3] = pose_out[:3]
            adjusted_pose[3:] = pose_out[3:]

            # Append the adjusted pose to the initial guess
            initial_guess.append(adjusted_pose)

        return np.concatenate([pose.flatten() for pose in initial_guess])


    def optimize_trust_constr(self):
        initial_guess = self.construct_initial_guess()

        if len(initial_guess) < 6 * len(self.poses):
            print("Not enough variables for optimization.")
            return

        def callback(xk, _):
            ccurrent_poses = xk.reshape((-1, 6))
            current_chi_squared = self.cost_function(xk)
            print("Current chi-squared:", current_chi_squared)
            # print("Current cost:", self.cost_function(xk))
            # print("Current poses:", current_poses)

        result = minimize(self.cost_function, initial_guess, method='trust-constr', jac=self.gradient, hess=self.hessian,
                  bounds=[(-np.inf, np.inf)] * len(initial_guess),
                  options={'maxiter': 2000, 'disp': True, 'gtol': 1e-8, 'xtol': 1e-8}, callback=callback)

        if result.success:
            optimized_poses = result.x.reshape((-1, 6))
            # Ensure the number of poses remains consistent
            if len(optimized_poses) == len(self.poses):
                self.poses = list(zip(range(len(optimized_poses)), optimized_poses))
                chi_squared = self.cost_function(result.x)
                print("Optimization successful. Chi-squared:", chi_squared)
            else:
                print("Number of poses changed during optimization.")
        else:
            print("Optimization failed.")
            print(result.message)

    def cost_function(self, flattened_poses):
        poses = flattened_poses.reshape((-1, 6))
        chi_squared = 0.0

        for edge in self.edges:
            id_out, id_in, measurement, info_matrix = edge
            pose_out, pose_in = self.get_poses_from_ids(id_out, id_in, poses)

            delta_pose = pose_in - pose_out
            predicted_measurement = np.linalg.norm(delta_pose)

            error = measurement - predicted_measurement
            info_matrix_inv = np.linalg.inv(info_matrix)

            chi_squared += error.T @ info_matrix_inv @ error

        for loop_closure in self.loop_closures:
            pose_index1, pose_index2, measurement, info_matrix = loop_closure
            pose1, pose2 = self.get_poses_from_ids(pose_index1, pose_index2, poses)

            delta_pose = pose2 - pose1
            predicted_measurement = np.linalg.norm(delta_pose)

            error = measurement - predicted_measurement
            info_matrix_inv = np.linalg.inv(info_matrix)

            chi_squared += error.T @ info_matrix_inv @ error

        return chi_squared


    def gradient(self, flattened_poses):
        poses = flattened_poses.reshape((-1, 6))
        gradient = np.zeros_like(flattened_poses)

        print("Gradient function called.")

        for edge in self.edges:
            id_out, id_in, measurement, info_matrix = edge
            pose_out, pose_in = self.get_poses_from_ids(id_out, id_in, poses)

            delta_pose = pose_in - pose_out
            predicted_measurement = np.linalg.norm(delta_pose)

            error = measurement - predicted_measurement
            info_matrix_inv = np.linalg.inv(info_matrix)

            J = self.compute_jacobian(pose_out, pose_in)

            # Corrected the shape mismatch in the matrix multiplication
            gradient[id_out * 6: id_out * 6 + 6] -= J[:, :6].T @ info_matrix_inv @ error
            gradient[id_in * 6: id_in * 6 + 6] += J[:, :6].T @ info_matrix_inv @ error

        # print("Gradient for edge:", gradient[id_out * 6: id_out * 6 + 6])
        # print("Gradient for edge:", gradient[id_in * 6: id_in * 6 + 6])

        for loop_closure in self.loop_closures:
            pose_index1, pose_index2, measurement, info_matrix = loop_closure
            pose1, pose2 = self.get_poses_from_ids(pose_index1, pose_index2, poses)

            delta_pose = pose2 - pose1
            predicted_measurement = np.linalg.norm(delta_pose)

            error = measurement - predicted_measurement
            info_matrix_inv = np.linalg.inv(info_matrix)

            J = self.compute_jacobian(pose1, pose2)

            gradient[pose_index1 * 6: pose_index1 * 6 + 6] -= J[:, :6].T @ info_matrix_inv @ error
            gradient[pose_index2 * 6: pose_index2 * 6 + 6] += J[:, :6].T @ info_matrix_inv @ error

        # print("Gradient for loop closure:", gradient)

        return gradient

    def hessian(self, flattened_poses):
        poses = flattened_poses.reshape((-1, 6))
        hessian = np.zeros((len(flattened_poses), len(flattened_poses)))

        print("Hessian function called.")

        for edge in self.edges:
            id_out, id_in, measurement, info_matrix = edge
            pose_out, pose_in = self.get_poses_from_ids(id_out, id_in, poses)

            delta_pose = pose_in - pose_out
            predicted_measurement = np.linalg.norm(delta_pose)

            error = measurement - predicted_measurement
            info_matrix_inv = np.linalg.inv(info_matrix)

            # Fix indexing here
            J = self.compute_jacobian(pose_out[:3], pose_in[:3])
            hessian[id_out * 6: id_out * 6 + 6, id_out * 6: id_out * 6 + 6] -= J[:, :6].T @ info_matrix_inv @ J[:, :6]
            hessian[id_in * 6: id_in * 6 + 6, id_in * 6: id_in * 6 + 6] -= J[:, :6].T @ info_matrix_inv @ J[:, :6]
            hessian[id_out * 6: id_out * 6 + 6, id_in * 6: id_in * 6 + 6] += J[:, :6].T @ info_matrix_inv @ J[:, :6]
            hessian[id_in * 6: id_in * 6 + 6, id_out * 6: id_out * 6 + 6] += J[:, :6].T @ info_matrix_inv @ J[:, :6]

        for loop_closure in self.loop_closures:
            pose_index1, pose_index2, measurement, info_matrix = loop_closure
            pose1, pose2 = self.get_poses_from_ids(pose_index1, pose_index2, poses)

            delta_pose = pose2 - pose1
            predicted_measurement = np.linalg.norm(delta_pose)

            error = measurement - predicted_measurement
            info_matrix_inv = np.linalg.inv(info_matrix)

            # Fix indexing here as well
            J = self.compute_jacobian(pose1[:3], pose2[:3])
            hessian[pose_index1 * 6: pose_index1 * 6 + 6, pose_index1 * 6: pose_index1 * 6 + 6] -= J[:, :6].T @ info_matrix_inv @ J[:, :6]
            hessian[pose_index2 * 6: pose_index2 * 6 + 6, pose_index2 * 6: pose_index2 * 6 + 6] -= J[:, :6].T @ info_matrix_inv @ J[:, :6]
            hessian[pose_index1 * 6: pose_index1 * 6 + 6, pose_index2 * 6: pose_index2 * 6 + 6] += J[:, :6].T @ info_matrix_inv @ J[:, :6]
            hessian[pose_index2 * 6: pose_index2 * 6 + 6, pose_index1 * 6: pose_index1 * 6 + 6] += J[:, :6].T @ info_matrix_inv @ J[:, :6]

        return hessian

    def compute_jacobian(self, pose_out, pose_in):
        J = np.zeros((6, 12))

        if pose_out.ndim == 1:
            R_i = pose_out[:3]
            t_i = pose_out[3:]
        else:
            R_i = pose_out[:3, :3]
            t_i = pose_out[:3, 3]

        if pose_in.ndim == 1:
            R_j = pose_in[:3]
            t_j = pose_in[3:]
        else:
            R_j = pose_in[:3, :3]
            t_j = pose_in[:3, 3]

        # Correct the assignment of values in the Jacobian
        J[:3, :3] = -np.eye(3)
        J[3:, 3:6] = -np.eye(3)
        J[:3, 6:9] = np.eye(3)
        J[3:, 9:] = np.eye(3)

        return J

    def visualize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        poses_array = np.array([pose for _, pose in self.poses])
        ax.scatter(poses_array[:, 0], poses_array[:, 1], poses_array[:, 2], label='Poses', marker='o', s=0.35, color='red')

        landmarks_array = np.array([landmark for _, landmark in self.landmarks])
        ax.scatter(landmarks_array[::3], landmarks_array[1::3], landmarks_array[2::3], label='Landmarks', marker='^')

        ax.plot(poses_array[:, 0], poses_array[:, 1], poses_array[:, 2], label='Optimized Trajectory', color='blue')

        for edge in self.edges + self.loop_closures:
            id_out, id_in, _, _ = edge
            pose_out, pose_in = self.get_poses_from_ids(id_out, id_in, poses_array)

            # ax.plot([pose_out[0], pose_in[0]], [pose_out[1], pose_in[1]], [pose_out[2], pose_in[2]],  color='gray', linestyle='dashed', alpha=0.5)

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        plt.show()

def parse_vertex3(line):
    _, vertex_id, *pose_values = line.split()
    vertex_id = int(vertex_id)
    pose = list(map(float, pose_values))
    return vertex_id, pose

def parse_landmark3(line):
    _, landmark_id, *landmark_values = line.split()
    landmark_id = int(landmark_id)
    landmark = list(map(float, landmark_values))
    return landmark_id, landmark

def parse_edge3(line):
    _, id_out, id_in, *measurement_and_info = line.split()
    id_out, id_in = int(id_out), int(id_in)
    measurement = list(map(float, measurement_and_info[:6]))
    info_matrix = np.zeros((6, 6))
    info_values = list(map(float, measurement_and_info[6:]))
    info_matrix[np.triu_indices(6)] = info_values
    return id_out, id_in, measurement, info_matrix

def parse_loop_closure(line):
    _, pose_index1, pose_index2, *measurement_and_info = line.split()
    pose_index1, pose_index2 = int(pose_index1), int(pose_index2)
    measurement = list(map(float, measurement_and_info[:6]))
    info_matrix = np.zeros((6, 6))
    info_values = list(map(float, measurement_and_info[6:]))
    info_matrix[np.triu_indices(6)] = info_values
    return pose_index1, pose_index2, measurement, info_matrix

def pose_difference(original_poses, optimized_poses):
    original_poses_array = np.array([pose[1] for _, pose in original_poses])
    optimized_poses_array = np.array([pose[1] for _, pose in optimized_poses])
    return np.linalg.norm(original_poses_array - optimized_poses_array)

# Example usage:
filename = 'parking-garage.txt'  # Replace with your actual dataset filename
slam = GraphSLAM()
slam.read_dataset(filename)

# Store the original poses before optimization
original_poses = list(slam.poses)

# Optimize using Trust Construtor
slam.optimize_trust_constr()
optimized_poses_trust_constr = list(slam.poses)

# Calculate and print the difference
difference_trust_constr = pose_difference(original_poses, optimized_poses_trust_constr)
print(f"Difference after Trust Construtor optimization: {difference_trust_constr}")

# Visualize the Trust Construtor optimization
slam.visualize()