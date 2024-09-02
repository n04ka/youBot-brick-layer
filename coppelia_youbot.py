from copy import deepcopy
from dataclasses import dataclass
import struct
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from math import pi as PI
from math import (atan, atan2, acos, sin, cos, prod)
import numpy as np
from time import sleep
from multiprocessing import Lock


class Sim:
    '''Represents sim. Use Sim.sim.{command}() to send commands to sim.'''
    
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    client.setStepping(False)
    verbose = True
    lock = Lock()


class Angle:
    '''Stores angles within the range of (-PI, PI], useful for orientation.'''
    
    def __init__(self, number: float | int = 0, is_degrees: bool = False) -> None:
        '''Creates angle object from both radians (default) or degrees.'''
        
        if is_degrees:
            number *= PI / 180
        self.value = float(number)
         
    
    @staticmethod
    def normalize(number: float | int) -> float:
        '''Cuts the value to fit the range (-PI, PI].'''
        
        while number <= -PI:
            number += 2*PI
        while number > PI:
            number -= 2*PI
        return number
    
    
    @property     
    def value(self) -> float:
        '''Gets the angle in radians.'''
        
        return self._value
    
    
    @value.setter
    def value(self, number: float | int) -> None:
        '''Setter, normalizes the input.'''
        
        self._value = Angle.normalize(number)
    
    
    def degrees(self) -> float:
        '''Gets the angle in degrees.'''
        
        return self.value / PI * 180
        
        
    def __str__(self) -> str:
        '''Useful for printing.'''
        
        return str(self.value)
    
    
    def __float__(self) -> float:
       '''Extracts the float value, similar to value getter.'''
        
       return self.value
   
   
    def __add__(self, other: "Angle") -> "Angle":
        return Angle(self.value + other.value)
    
    
    def __sub__(self, other: "Angle") -> "Angle":
        return Angle(self.value - other.value)


class CylinderCoords:
    '''Represents a point in cylinder coorinates.'''
        
    def __init__(self, r: float, h: float, alpha: Angle) -> None:
        '''Creates a point in cylinder coorinates.'''
        
        self.r = r
        self.h = h
        self.alpha = alpha
        
        
    @property
    def alpha(self) -> float:
        '''Extracts float alpha from Angle attribute.'''
        
        return self._alpha.value
    
    
    @alpha.setter
    def alpha(self, value: float | Angle) -> None:
        '''Sets alpha converting it to Angle if float is given.'''
        
        self._alpha = value if isinstance(value, Angle) else Angle(value)
        
    
    def __repr__(self) -> str:
        '''Useful for printing.'''
        
        return f'\tr:\t{self.r:.2f}\n\th:\t{self.h:.2f}\n\talpha:\t{self.alpha:.2f}'

@dataclass
class Pose:
    '''Represents position and orientation of something.'''
    
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    
    
    def get_pos(self) -> tuple[float, float, float]:
        '''Gets x, y, z.'''
        
        return self.x, self.y, self.z
    
    
    def get_euler(self) -> tuple[float, float, float]:
        '''Gets alpha, beta, gamma.'''
        
        return self.alpha, self.beta, self.gamma


class SimObject:
    '''Represents sim object from CoppeliaSim.'''
    
    def __init__(self, path: str) -> None:
        '''Creates sim object.'''
        
        with Sim.lock:
            self._obj = Sim.sim.getObject(path)  
    
    
    def get_pose(self) -> Pose:
        '''Gets the pose of the object'''
        
        return Pose(*self.get_pos(), *self.get_orient())
    
        
    def get_xy(self) -> tuple[float, float]:
        '''Gets the X and Y decart coordinates of the object.'''
        
        return self.get_pos()[:2]
    
    
    def get_pos(self) -> tuple[float, float, float]:
        '''Gets the decart coordinates of the object.'''
        
        with Sim.lock:
            coords = Sim.sim.getObjectPosition(self._obj, Sim.sim.handle_world)
        return coords
    
    
    def get_orient(self) -> tuple[float, float, float]:
        '''Gets euler angles.'''
        
        with Sim.lock:
            return Sim.sim.getObjectOrientation(self._obj, Sim.sim.handle_world)

    
    def get_Tait_Bryan(self) -> tuple[float, float, float]:
        '''Gets Tait_Bryan angles.'''
        euler = self.get_orient()
        with Sim.lock:
            return Sim.sim.alphaBetaGammaToYawPitchRoll(*euler)
        
        
    def get_matrix(self) -> np.ndarray:
        with Sim.lock:
            matrix = np.array(Sim.sim.getObjectMatrix(self._obj, Sim.sim.handle_world)).reshape((3, 4))
        return np.vstack((matrix, np.array([0., 0., 0., 1.])))
    

    def get_azimuth(self) -> float:
        '''Gets yaw.'''
        
        return float(Angle(self.get_Tait_Bryan()[2]-PI))


class VisionSensor(SimObject):
    '''Represents a vision sensor'''
    
    def __init__(self, path: str, fov: float, shape: tuple[int, int], near: float, far: float) -> None:
        super().__init__(path)
        self.fov = fov
        self.near = near
        self.far = far
        self.shape = shape
        self.center = np.array(shape)/2
        self.focus = shape[1]/2/np.tan(self.fov/2) # in pixels
        
    
    def shoot(self, do_rgb: bool = False) -> tuple:
        matrix = self.get_matrix()
        
        with Sim.lock:
            d, res = Sim.sim.getVisionSensorDepth(self._obj)
        assert tuple(res) == self.shape
        d = struct.unpack(f'{prod(res)}f', d)
        depth = np.array(d).reshape(res)
        if res[0] == res[1]:
            depth = depth.transpose()
        
        if not do_rgb:
            return matrix, depth
            
        with Sim.lock:
            img, res = Sim.sim.getVisionSensorImg(self._obj)
        assert tuple(res) == self.shape
        img = np.array(list(img), dtype=np.uint8).reshape(*res, 3).transpose((1, 0, 2))
        # import matplotlib.pyplot as plt
        # plt.imshow(img)
        # plt.show()
                
        return matrix, depth, img
        
    
    
    def filter_depth(self, depth: np.ndarray) -> np.ndarray:
        depth[depth > 0.99] = np.nan
        return depth * (self.far - self.near) + self.near
    
    
    def convert_from_uvd(self, u: np.ndarray, v: np.ndarray, d: np.ndarray) -> np.ndarray:
        # print(u.shape, v.shape)
        # import matplotlib.pyplot as plt
        # plt.imshow(d)
        # plt.show()
        x = ((u - self.center[1]) * d / self.focus).flatten()
        y = ((v - self.center[0]) * d / self.focus).flatten()
        z = d.flatten()
        # print(x.shape, y.shape, z.shape)
        return np.array([x, y, z])
    
    
    def camera2global(self, coords: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        x, y, z = coords
        global_cloud = np.array([-y, x, z, np.ones_like(x)])
        # import matplotlib.pyplot as plt
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(*global_cloud[:3], s=0.01)
        # ax.set_aspect('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.title('Облако в системе координат лидара с изменёнными осями')
        # plt.show()
        rotated = np.dot(matrix, global_cloud)[:3]
        return rotated

    
    def get_cloud(self, img: np.ndarray, matrix: np.ndarray, topleft: tuple | None = None, bottomright: tuple | None = None) -> np.ndarray:
        if topleft is not None and bottomright is not None:
            depth = self.filter_depth(img[topleft[1]:bottomright[1], topleft[0]:bottomright[0]])
            u = np.arange(topleft[0], bottomright[0]).reshape((1, -1))
            v = np.arange(topleft[1], bottomright[1]).reshape((-1, 1))
        else:
            depth = self.filter_depth(img)
            u = np.arange(self.shape[1]).reshape((1, -1))
            v = np.arange(self.shape[0]).reshape((-1, 1))
        # import matplotlib.pyplot as plt
        # plt.imshow(depth)
        # plt.show()
        # print('shape: ', self.shape)
        local_cloud = self.convert_from_uvd(u, v, depth)
        # plt.scatter(*local_cloud[[0, 2]], s=0.1)
        # plt.gca().set_aspect('equal')
        # plt.title('Облако в системе координат лидара')
        # plt.show()
        global_cloud = self.camera2global(local_cloud, matrix)
        # import matplotlib.pyplot as plt
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(*global_cloud, s=0.01)
        # ax.set_aspect('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.title('Облако в глобальной системе координат')
        # plt.show()
        # return global_cloud[:, ~np.isnan(global_cloud).any(axis=0)]
        return global_cloud
    

class ArmJoint(SimObject):
    '''Represents a joint of manipulator.'''
    
    def __init__(self, path: str, length: float = 0.0) -> None:
        '''Creates a joint, length can be skip if does not matter.'''
        
        super().__init__(path)
        self._length = length
        self._target: float
        
    
    @property
    def q(self) -> float:
        '''Gets the actual joint angle from sim.'''
        
        with Sim.lock:
            return Sim.sim.getJointPosition(self._obj)
    
    
    @q.setter
    def q(self, value: float) -> None:
        '''Sets the desired joint angle.'''
        
        self.target = value
        with Sim.lock:
            Sim.sim.setJointTargetPosition(self._obj, value)
        
    
    def is_act_finished(self, precision: float = 0.01) -> bool:
        '''Checks if target matches actual angle in sim.'''
        
        return abs(self.q - self.target) < precision
        

class Gripper(SimObject):
    '''Represents manipulator's gripper.'''
    
    
    def __init__(self, path: str) -> None:
        '''Creates gripper object and sets its state to opened.
        Does not use the reference which is for debug only.'''
        
        super().__init__(path)
        with Sim.lock:
            self._script = Sim.sim.getScript(Sim.sim.scripttype_childscript, self._obj)
        self.release()


    def grab(self, wait: bool = True) -> None:
        '''Sends the grab command.'''
        
        self._state = 'closed'
        with Sim.lock:
            Sim.sim.callScriptFunction('grab', self._script)
        if wait:
            sleep(1)
                
    
    def release(self, wait: bool = True) -> None:
        '''Sends the release command.'''
        
        self._state = 'opened'
        with Sim.lock:
            Sim.sim.callScriptFunction('drop', self._script)
        if wait:
            sleep(1)
        
        
    def is_opened(self) -> bool:
        '''Checks the assigned state (not actual).'''
        
        return self._state == 'opened'


class Platform:
    '''Represents the youBot 4-wheeled platform.'''
    
    speed = 7
    rotation_speed = 3
    precision = 0.05
    
    
    def __init__(self, wheels: list[SimObject], reference: SimObject) -> None:
        '''Creates a platform object from wheels and reference.'''
        
        self._wheels = wheels
        self.ref = reference
        self.stop()
       
        
    def set_movement(self, fb_vel: float = 0, lr_vel: float = 0, rot_vel: float = 0, vector: tuple[float, float, float] | None = None) -> None:
        '''Sends the desired speeds to sim.'''
        
        if vector is not None:
            fb_vel, lr_vel, rot_vel = vector
    
        velocities = [- fb_vel - lr_vel - rot_vel,
                      - fb_vel + lr_vel - rot_vel,
                      - fb_vel - lr_vel + rot_vel,
                      - fb_vel + lr_vel + rot_vel]
        
        
        with Sim.lock:
            for wheel, vel in zip(self._wheels, velocities):
                Sim.sim.setJointTargetVelocity(wheel._obj, vel)
            
    
    def stop(self) -> None:
        '''Sets all speeds to 0.'''
        
        self.set_movement(0, 0, 0)
    
    
    def get_back(self) -> None:
        '''Gets back for half a second.'''
        
        self.set_movement(-self.speed, 0, 0)
        sleep(0.5)
        self.stop()
    
      
    def rotate_to(self, angle: Angle = Angle(0), point: tuple | None = None) -> None:
        '''Rotates the platform to the desired angle or towards the decart point. Waits till the maneuver is done.'''
        
        if point is not None:
            angle = get_direction(self.ref.get_pos(), point)
        
        while not is_azimuth_reached(self.ref, angle, Platform.precision):
            self.set_movement(0, 0, self.get_ang_vel(angle))
        self.stop()
       
       
    def move_to(self, point: tuple) -> None:
        '''Moves the platform to the desired point without rotation. Waits till the maneuver is done.'''
        
        while not is_point_reached(self.ref, point, True, Platform.precision):
            self.set_movement(*self.get_lin_vel(point), 0.0)
        self.stop()
        
        
    def travel_to(self, point: tuple) -> None:
        '''Rotates the platform towards the given point, then moves it straight to the point'''
        
        self.rotate_to(get_direction(self.ref.get_pos(), point))
        self.move_to(point)
    
        
    def get_ang_vel(self, angle: Angle = Angle(0), point: tuple | None = None) -> float:
        '''Calculates angular velocity to reach the desired angle, chooses the shortest way.'''
        
        if point is not None:
            angle = get_direction(self.ref.get_pos(), point)
            
        if is_azimuth_reached(self.ref, angle, Platform.precision):
            return 0.0
        
        delta = angle.value - self.ref.get_azimuth()
        speed = Platform.speed * np.sign(delta) * (-1 if abs(delta) <= PI else 1)
        
        return speed
        
        
    def get_lin_vel(self, point: tuple) -> tuple[float, float]:
        '''Calculates linear velocities to reach the desired point.'''
        
        if is_point_reached(self.ref, point, True, Platform.precision):
            return 0.0, 0.0
        
        direction = self.ref.get_azimuth() - get_direction(self.ref.get_pos(), point).value
        x_vel = Platform.speed * cos(direction)
        y_vel = Platform.speed * sin(direction)
        return x_vel, y_vel
      

class Arm:
    '''Represents the whole manipulator.'''    
    
    park_pose = (0.0, PI/4, PI/4, PI/2, 0.0)
    precision = 0.01
    r_base_shift = 0.033
    h_base_shift = 0.26
    max_grab_distance = 0.4
    
    
    def __init__(self, links: list[ArmJoint], gripper: Gripper, reference: SimObject) -> None:
        '''Creates a manupulator from joints, gripper and mount point reference.''' 
        
        if len(links) != 5:
            raise Warning(f'Expected list[ArmJoint] of length 5, got length: {len(links)}')
        
        self._links = links
        self.gripper = gripper
        self.ref = reference
        

    @property
    def q(self) -> list[float]:
        '''Gets the vector of actuall joint states.'''
        
        return [link.q for link in self._links]
    
    
    @q.setter
    def q(self, vector: tuple | list[Angle]) -> None:
        '''Sets the vector of target angles.'''
        
        for link, q in zip(self._links, vector):
            link.q = float(q)
    
    
    def go_until(self, vector: tuple | list[Angle]) -> None:
        '''Sets the vector of target angles and waits the completion.'''
        
        self.q = vector
        self.wait()
            

    def is_act_finished(self) -> bool:
        '''Checks if all joints had reached their targets.'''
        
        return all([link.is_act_finished(Arm.precision) for link in self._links])
    
    
    def wait(self) -> None:
        '''Waits until all joints reach their targets.'''
        
        while not self.is_act_finished():
            pass
    
    
    def park(self, wait: bool = True) -> None:
        '''Parks the manipulator into Arm.park_pose state, waits for completion by default.'''
        
        self.q = Arm.park_pose
        if wait:
            self.wait()
    
    
    def decart_to_local(self, coords: tuple[float, float, float]) -> CylinderCoords:
        '''Calculates a decart point into the arm coordinate system.'''
        
        local = from_decart((*self.ref.get_xy(), Arm.h_base_shift), coords)
        local.r += Arm.r_base_shift
        local.alpha = Angle(local.alpha - self.ref.get_azimuth())
        return local
    
    
    def calculate_q(self, local_coords: CylinderCoords, target_rotation: Angle) -> list[Angle] | None:
        '''Calculates the vector of q to get to the target point considering target rotation.
        Assumes that target object is symmetric. Returns None if target is unreachable.'''
        
        L = [link._length for link in self._links]
        H = local_coords.h + L[4]
        R = (local_coords.r**2 + H**2)**0.5
        
        if R > L[2] + L[3]:
            print(f'R: {R:.2f}, L2+L3: {L[2] + L[3]:.2f}')
            raise Warning('Object out of manipulator range')
            return None
        
        BETA = acos((L[2]**2 + R**2 - L[3]**2) / (2*L[2]*R))
        
        q = [local_coords.alpha,
             PI/2 - atan(H/abs(local_coords.r)) - BETA, # type: ignore
             PI - acos((L[3]**2 + L[2]**2 - R**2) / (2*L[2]*L[3]))]
        q.append(PI - q[1] - q[2])
        
        gripper_angle = Angle(PI + q[0] - target_rotation.value + self.ref.get_azimuth())
        if gripper_angle.value > PI/2:
            gripper_angle.value -= PI
        elif gripper_angle.value < -PI/2:
            gripper_angle.value += PI
        
        q.append(gripper_angle.value)
        
        if local_coords.r < 0:
            for i in range(1, 4):
                q[i] *= -1
            
        return [Angle(value) for value in q]
        
    
    def move_to(self, coords: tuple[float, float, float] | CylinderCoords, target_orient: Angle, wait: bool = True) -> None:
        '''Moves the manipulator to the target point considering target rotation.
        Assumes that target object is symmetric. Does nothing if point is unreachable.'''
        
        is_cyl = isinstance(coords, CylinderCoords)
        q = self.calculate_q(coords if is_cyl else self.decart_to_local(coords), target_orient)
            
        if q is not None:
            self.q = q
        else:
            return  
        
        if wait:
            while not self.is_act_finished():
                q = self.calculate_q(coords if is_cyl else self.decart_to_local(coords), target_orient)
                if q is None:
                    return
                # print(''.join([f'q{i}: {_.value:.2f}\t' for i, _ in enumerate(q)]))
                self.q = q


    def put(self, target: tuple[float, float, float] | CylinderCoords, target_orient: Angle, dist_above_target: float = 0.04) -> None:
        '''Puts object in the given coordinates.'''
        
        is_cyl = isinstance(target, CylinderCoords)
        
        shift = dist_above_target
        while shift > -0.01:
            if is_cyl:
                coords = deepcopy(target)
                coords.h += shift
            else:
                x, y, z = target
                z += shift
                coords = (x, y, z)
                
            self.move_to(coords, target_orient)
            shift -= 0.01
            
        self.gripper.release()
        
        shift = 0.01
        while shift < dist_above_target:
            if is_cyl:
                coords = deepcopy(target)
                coords.h += shift
            else:
                x, y, z = target
                z += shift
                coords = (x, y, z)
            self.move_to(coords, target_orient)
            shift += 0.01


    def take(self, target: tuple[float, float, float] | CylinderCoords, target_orient: Angle, dist_above_target: float = 0.04) -> None:
        '''Takes object in the given coordinates.'''
        
        is_cyl = isinstance(target, CylinderCoords)
        
        shift = dist_above_target
        while shift > -0.01:
            if is_cyl:
                coords = deepcopy(target)
                coords.h += shift
            else:
                x, y, z = target
                z += shift
                coords = (x, y, z)
                
            self.move_to(coords, target_orient)
            shift -= 0.01
            
        self.gripper.grab()
    

class YouBot:
    '''Represents youBot.'''
    
    def __init__(self, arm: Arm, platform: Platform) -> None:
        '''Creates the robot from manipulator and platform.'''
        
        self.arm = arm
        self.platform = platform
    
    
    def approach(self, target: Pose | tuple[float, float]) -> None:
        '''Approaches the target for the arm to be able to move there.'''
        
        if isinstance(target, Pose):
            X, Y = target.x, target.y
        else:
            X, Y = target
        
        robot_coords = self.platform.ref.get_xy()
        R = get_dist(robot_coords, (X, Y))
        pickup_point = (X - 0.9*self.arm.max_grab_distance/R * (X - robot_coords[0]),
                        Y - 0.9*self.arm.max_grab_distance/R * (Y - robot_coords[1]))
        
        if Sim.verbose:
            print(f'Pickup point: {pickup_point[0]:.2f}, {pickup_point[1]:.2f}')
            
        self.platform.travel_to(pickup_point)
        self.platform.rotate_to(point=pickup_point)
    
    
    def go_pick(self, target: SimObject):
        '''Robot approaches the target and picks it'''
        
        self.approach(target.get_pose())
        self.arm.take(target.get_pos(), Angle(target.get_azimuth()))
        self.arm.park(False)
    
    
    def go_place(self, target: Pose) -> None:
        '''Robot approaches the target, places a brick to it.'''

        self.approach(target)
        self.arm.put(target.get_pos(), Angle(target.gamma))
        self.arm.park(False)
               

def from_decart(center: tuple[float, float, float], target: tuple[float, float, float]) -> CylinderCoords:
    '''Transforms target decart point into the cyliner system with the given center.'''
    
    r = get_dist(center, target, xy_only=True)
    h = target[2] - center[2]
    alpha = get_direction(center, target)
    return CylinderCoords(r, h, alpha)


def get_dist(point1: tuple, point2: tuple, xy_only: bool = False) -> float:
    '''Gets the distance between two n-dismentional points. Considers the lowest if dismentions differ.'''
    
    limit = 2 if xy_only else len(point1)+1
    return sum([(_ - __)**2 for _, __ in zip(point1[:limit], point2)])**0.5


def get_direction(start: tuple, finish: tuple) -> Angle:
    '''Gets the direction from the start decart point to the finish one.'''
    
    dx = finish[0] - start[0]
    dy = finish[1] - start[1]
    return Angle(atan2(dy, dx))


def assemble_robot(root: str = '/youBot/') -> YouBot:   
    '''Inits my youBot.'''
    
    ref_platform = SimObject(f'{root}youBot_ref')
    wheels = [SimObject(f'{root}rollingJoint_{id}') for id in ['fl', 'rl', 'rr', 'fr']]
    platform = Platform(wheels, ref_platform)
    
    lengths = [0.0, 0.0, 0.155, 0.135, 0.218]
    ref_arm = SimObject(f'{root}arm_ref')
    gripper = Gripper('./Rectangle7')
    joints = [ArmJoint(f'{root}youBotArmJoint{i}', lengths[i]) for i in range(5)]
    arm = Arm(joints, gripper, ref_arm)
    return YouBot(arm, platform)


def is_point_reached(ref: SimObject, point: tuple, xy_only: bool = False, precision: float = 0.05) -> bool:
    '''Checks if the reference reached the given decart point. Considers only XOY projection if xy_only.'''
    
    coords = ref.get_pos()[:2] if xy_only else ref.get_pos()
    return all([abs(_ - __) < precision for _, __ in zip(coords, point)])


def is_azimuth_reached(ref: SimObject, angle: Angle, precision: float = 0.01) -> bool:
    '''Checks if the reference reached the given angle.'''
    
    # print(f'Azimuth delta: {abs(ref.get_azimuth() - angle.value)}')
    # print(f'Azimuth: {ref.get_azimuth()}')
    # print(f'Target: {angle.value}\n')
    return abs(ref.get_azimuth() - angle.value) < precision


def compare_poses(pose1: Pose, pose2: Pose) -> None:
    '''Displays the difference between two poses.'''
    
    print('Accuracy:')
    for attribute in pose1.__dict__:
        print(f"\t{attribute} delta:\t{pose1.__dict__[attribute]-pose2.__dict__[attribute]:.3f}")
    print(f'\tTotal shift:\t{get_dist(pose1.get_pos(), pose2.get_pos()):.4f}')

    