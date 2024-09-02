import cv2 as cv2
import numpy as np
from collections import Counter
from scipy.spatial.transform import Rotation


def rotate2D(origin: tuple[float, float], cloud: np.ndarray, angle: float) -> np.ndarray:
    assert cloud.shape[0] == 2, 'wrong input shape'

    ox, oy = origin
    px, py = cloud

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.array([qx, qy])


def dist_points2line(points: np.ndarray, line: tuple[float, float, float]) -> np.ndarray:
    assert points.shape[0] == 2, 'wrong input shape'
    a, b, c = line
    return np.abs(a*points[0] + b*points[1] + c) / np.sqrt(a**2 + b**2)


def dist_points2plane(points: np.ndarray, plane: tuple[float, float, float, float]) -> np.ndarray:
    assert points.shape[0] == 3, 'wrong input shape'
    a, b, c, d = list(map(float, plane))
    return np.abs(a*points[0] + b*points[1] + c*points[2] + d) / np.sqrt(a**2 + b**2 + c**2)
    
    
def line_by_2_points(points) -> tuple[float, float, float]:
    assert len(points) == 2
    p1, p2 = points
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0]*p2[1] - p1[1]*p2[0]
    return a, b, c


def plane_by_3_points(points) -> tuple[float, float, float, float]:
    assert len(points) == 3
    p1, p2, p3 = points
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = - a * x1 - b * y1 - c * z1
    return a, b, c, d


def iteration2D(cloud: np.ndarray, precision: float = 0.01) -> tuple[int, tuple[float, float, float]]:
    seed = cloud[:, np.random.randint(0, cloud.shape[1]-1, size=2)].transpose()
    line = line_by_2_points(seed)
    d = dist_points2line(cloud, line)
    return np.count_nonzero(d <= precision), line


def iteration3D(cloud: np.ndarray, precision: float = 0.01) -> tuple[int, tuple[float, float, float, float]]:
    if cloud.shape[1] <= 2:
        return 0, (0, 0, 0, 0)
    seed = cloud[:, np.random.randint(0, cloud.shape[1]-1, size=3)].transpose()
    plane = plane_by_3_points(seed)
    d = dist_points2plane(cloud, plane)
    return np.count_nonzero(d <= precision), plane


def get_center3D(cloud: np.ndarray) -> np.ndarray:
    assert cloud.shape[0] == 3
    mins = np.min(cloud, axis=1)
    maxs = np.max(cloud, axis=1)
    return (maxs + mins) / 2


def is_on_plane(plane: tuple[float, float, float, float], points: np.ndarray, precision: float = 0.001) -> np.ndarray:
    return dist_points2plane(points, plane) < precision
    

def find_planes(cloud: np.ndarray, n_planes: int = 3, precision: float = 0.001, iterations: int = 100) -> list[tuple[float, float, float, float]]:
    np.seterr(all="ignore")
    planes = []
    
    step_cloud = cloud.copy()
    for _ in range(n_planes):
        d = dict(iteration3D(step_cloud, precision) for _ in range(iterations))
        planes.append(d[max(d)])
        step_cloud = step_cloud[:, ~is_on_plane(planes[-1], step_cloud)]
    return planes


def remove_non_orthogonal_planes(cloud: np.ndarray, planes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    colors = np.array([is_on_plane(plane, cloud) for plane in planes])
    priority = np.count_nonzero(colors, axis=1)
    normales = np.array(planes)[:, :3] / np.linalg.norm(np.array(planes)[:, :3], axis=1, keepdims=True)
    orthogonality = {(j, i) : np.abs(np.dot(normales[j], normales[i])) < .1 for j in range(0, 2) for i in range(j+1, 3)}
    conflicts = [key for key in orthogonality if ~orthogonality[key]]
    match len(conflicts):
        case 0:
            pass
        case 1:
            planes.pop(conflicts[0][0] if priority[conflicts[0][0]] < priority[conflicts[0][1]] else conflicts[0][1])
        case 2:
            for plane in conflicts[0]:
                if plane in conflicts[1]: 
                    planes.pop(plane)
                    break
        case _:
            raise RuntimeError('no orthogonal faces')
    return planes


def construct_basis(cloud: np.ndarray, planes: list[tuple[float, float, float, float]], brick_dims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = np.array([is_on_plane(plane, cloud) for plane in planes])
    actual_cloud_i = np.any(colors, axis=0)
    center = get_center3D(cloud[:, actual_cloud_i])
    dims = [2*dist_points2plane(center, plane) for plane in planes]
    fitting_faces = [np.argmin(np.abs(brick_dims - dim)) for dim in dims]
    conflict, count = Counter(fitting_faces).most_common(1)[0]
    match count:
        case 3:
            raise ValueError(f'All faces pretend to be {conflict}')
        case 2:
            i = len(fitting_faces) - 1 - fitting_faces[::-1].index(conflict)
            fitting_faces[i] = (set(range(3)) - set(fitting_faces)).pop() # type: ignore
        case _:
            pass

    basis = [[]]*3
    for face, plane in zip(fitting_faces, planes):
        basis[face] = plane[:3] # type: ignore
    match len(fitting_faces):
        case 1:
            raise ValueError(f'Not enough faces to create a basis: {len(fitting_faces)}')
        case 2:
            missing_dim = (set(range(3)) - set(fitting_faces)).pop() # type: ignore
            match missing_dim:
                case 2:
                    basis[missing_dim] = list(np.cross(basis[0][:3], basis[1][:3]))
                case 1:
                    basis[missing_dim] = list(np.cross(basis[2][:3], basis[0][:3]))
                case 0:
                    basis[missing_dim] = list(np.cross(basis[1][:3], basis[2][:3]))
                case _:
                    raise ValueError('Unexpected dim')
                
    basis = np.array(basis).transpose()
    norm_basis = basis / np.linalg.norm(basis, axis=0)
    if norm_basis[2, 2] < 0: 
        norm_basis *= -1
    a, b, c = norm_basis.transpose()
    if np.dot(a, np.cross(b, c)) < 0: # переход к правой тройке
        norm_basis = np.array([a, -b, c]).transpose()
        
    return center, norm_basis


def get_orientation(basis: np.ndarray) -> np.ndarray:
    global_basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rot_matrix = np.dot(np.linalg.inv(basis), global_basis)
    r = Rotation.from_matrix(rot_matrix)
    return r.as_euler("xyz")