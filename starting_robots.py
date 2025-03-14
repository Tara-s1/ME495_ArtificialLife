import random

def robot_four():
    robot_data = []
    robot_data.append((0, 0.1, 0.3, 0.1, -1, -1))
    robot_data.append((0.0, 0.03, 0.03, 0.07, 0, -1))
    robot_data.append((0.05, 0.0, 0.05, 0.1, 1, -1))
    robot_data.append((0.2, 0.0, 0.04, 0.11, 2, -1))
    robot_data.append((0.27, 0.0, 0.02, 0.1, 3, -1))
    return robot_data, 4

def robot_three():
    robot_data = []
    robot_data.append((0, 0.1, 0.3, 0.1, -1, -1))
    robot_data.append((0.01, 0.0, 0.02, 0.1, 0, -1))
    robot_data.append((0.14, 0.0, 0.05, 0.1, 1, -1))
    robot_data.append((0.22, 0.0, 0.03, 0.1, 2, -1))
    return robot_data, 3

def robot_five():
    robot_data = []
    robot_data.append((0, 0.1, 0.3, 0.1, -1, -1))
    robot_data.append((0.0, 0.03, 0.04, 0.07, 0, -1))
    robot_data.append((0.05, 0.0, 0.03, 0.1, 1, -1))
    robot_data.append((0.12, 0.0, 0.03, 0.1, 2, -1))
    robot_data.append((0.2, 0.02, 0.03, 0.08, 3, -1))
    robot_data.append((0.23, 0.0, 0.04, 0.1, 4, -1))
    return robot_data, 5


def robot_eight_uniform(): 
    robot_data = []
    robot_data.append((0, 0.1, 0.3, 0.1, -1, -1))
    robot_data.append((0.0, 0.03, 0.03, 0.07, 0, -1))
    robot_data.append((0.035, 0.03, 0.03, 0.07, 1, -1))
    robot_data.append((0.07, 0.03, 0.03, 0.07, 2, -1))
    robot_data.append((0.105, 0.03, 0.03, 0.07, 3, -1))
    robot_data.append((0.14, 0.03, 0.03, 0.07, 4, -1))
    robot_data.append((0.175, 0.03, 0.03, 0.07, 5, -1))
    robot_data.append((0.210, 0.03, 0.03, 0.07, 6, -1))
    robot_data.append((0.245, 0.03, 0.03, 0.07, 7, -1))
    return robot_data, 8