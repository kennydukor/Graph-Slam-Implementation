from Environment.grid_world import GridWorld
from Robot.robot import Robot
from Algorithms.graph_slam2 import GraphSLAM #, GraphSLAM3D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import copy
# from mayavi import mlab

TIME_STEP = 7 # time step
DISTANCE = 1
WORLD_SIZE = (100, 100)
NUM_LANDMARKS = 7
SENSOR_RANGE = 30


if __name__ == '__main__':

    # Create world
    world = GridWorld(world_size=WORLD_SIZE, num_landmarks=NUM_LANDMARKS, discrete=False)
    world.landmarks = world.get_landmarks()

    # Create robot
    robot = Robot(world, sensor_range=SENSOR_RANGE)

    # State Vector [x y yaw v]'
    true_pose = robot.state_space # x, y, yaw
    pose_DR = true_pose 
    # history initial values
    true_pose_store = true_pose
    pose_DR_store = pose_DR
    _, z, _, _ = robot.observe_landmarks(true_pose, pose_DR, control_input=np.array([[0,0]]).T)
    measurment_store = [z]

    for i in range(TIME_STEP):
        u = None
        if i <= TIME_STEP // 2:
            u = robot.control_input(v=7, w=np.deg2rad(10))
        else:
            u = robot.control_input(v=6.0, w=np.deg2rad(-8))
        true_pose, z, pose_DR, ud = robot.observe_landmarks(true_pose, pose_DR, u)
        pose_DR_store = np.hstack((pose_DR_store, pose_DR))
        true_pose_store = np.hstack((true_pose_store, true_pose))
        measurment_store.append(z)


    # visualize
    graphics_radius = 0.3
    for landmark in world.landmarks:
        plt.plot(landmark[0], landmark[1], "*", markersize=20)
    plt.plot(pose_DR_store[0, :], pose_DR_store[1, :], '.', markersize=50, alpha=0.8, label='Odom')
    plt.plot(true_pose_store[0, :], true_pose_store[1, :], '.', markersize=20, alpha=0.6, label='X_true')

    for i in range(pose_DR_store.shape[1]):
        x = pose_DR_store[0, i]
        y = pose_DR_store[1, i]
        yaw = pose_DR_store[2, i]
        plt.plot([x, x + graphics_radius * np.cos(yaw)],
                [y, y + graphics_radius * np.sin(yaw)], 'r')
        d = measurment_store[i][:, 0]
        angle = measurment_store[i][:, 1]
        plt.plot([x + d * np.cos(angle + yaw)], [y + d * np.sin(angle + yaw)], '.',
                markersize=20, alpha=0.7)
        plt.legend()
    plt.grid()
    plt.show()

    # slam
    C_SIGMA1 = 0.5
    C_SIGMA2 = 0.5
    C_SIGMA3 = np.deg2rad(1.0)
    slam = GraphSLAM(pose_DR_store, measurment_store, world.landmarks, graphics_radius=graphics_radius)
    slam.perform_slam(pose_DR_store, true_pose_store, len(robot.state_space), C_SIGMA1, C_SIGMA2, C_SIGMA3, number_of_iterations=30)
    slam.visualize_updated_pose(true_pose_store)