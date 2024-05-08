import numpy as np


def get_origin(l: float, w: float, h: float):
    return np.array([[ l/2,    w/2,    h/2],
                     [ l/2,    w/2,   -h/2],
                     [ l/2,   -w/2,    h/2],
                     [ l/2,   -w/2,   -h/2],
                     [-l/2,    w/2,    h/2],
                     [-l/2,   -w/2,   -h/2],
                     [-l/2,   -w/2,    h/2],
                     [-l/2,    w/2,   -h/2]])


def rotate(cloud: np.ndarray, angle: float):
    center: np.ndarray = np.average(cloud, axis=0)
    points = cloud - center
    rot_matrix = np.array([ [np.cos(angle), -np.sin(angle), 0.0],
                            [np.sin(angle), np.cos(angle), 0.0],
                            [0.0, 0.0, 1.0]])
    return np.dot(points.squeeze(), rot_matrix).squeeze() + center


def calc_error(cloud, origin):
    return np.array([np.min(np.linalg.norm(point-cloud, axis=1))**2 for point in origin]).mean()


def icp(points: np.ndarray, origin: np.ndarray, iterations: int):
    old_center = np.average(points, axis=0)
    cloud = points + np.average(origin, axis=0) - np.average(points, axis=0)
    error = np.inf
    for _ in range(iterations):
        closest_origin_points = origin[np.array([np.argmin(np.linalg.norm(point-origin, axis=1)) for point in cloud])]
        cloud += np.average(closest_origin_points, axis=0) - np.average(cloud, axis=0)
        angles = np.linspace(0.0, np.pi, 30)
        errors = np.array([calc_error(origin, rotate(cloud, angle)) for angle in angles])
        min_i = np.argmin(errors)
        cloud = rotate(cloud, angles[min_i])
        error = errors[min_i]
    
    new_center = np.average(cloud, axis=0)
    shift = old_center - new_center
    old_rotation = np.arctan2(old_center[1]-points[0, 1], old_center[0]-points[0, 0])
    new_rotation = np.arctan2(new_center[1]-cloud[0, 1], new_center[0]-cloud[0, 0])
    rotation = (new_rotation - old_rotation) % (2*np.pi)
    
    return shift, rotation


def get_corners(cloud):
    x0, y0, z0 = np.nanargmin(cloud, axis=1)
    x1, y1, z1 = np.nanargmax(cloud, axis=1)
    return np.array([cloud[:, i] for i in (x0, x1, y0, y1, z0, z1)])