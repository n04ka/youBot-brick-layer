from numpy import ndarray
import icp
from kuka import *
from time import time
from navigation import *
from schematics import *
from vision import *
from clusters import *
from ultralytics import YOLO
from itertools import cycle
from math import ceil
from multiprocessing import Queue, Process
from threading import Thread
import warnings
from enum import Enum



class Simulation(Sim):
    
    def __init__(self) -> None:
        self.output = Queue()
        self.proc = {
            'vision' : Process(target=Vision, args=(Sim.lock, self.output), daemon=True, name='VISION'),
            'lidar' : Process(target=Lidar, args=(Sim.lock, self.output), daemon=True, name='LIDAR'),
            'control' : Process(target=Control, args=(Sim.lock, self.output), name='CONTROL')
        }

    
    def start(self):
        with self.lock:
            Sim.sim.startSimulation()
            
        for proc in self.proc.values(): proc.start()
            
        self.proc['control'].join()
        self.proc['lidar'].terminate()
        self.proc['vision'].terminate()
        
        with self.lock:
            Sim.sim.stopSimulation()
        print('finished')
        

class Vision:
    
    def __init__(self, lock, output: Queue) -> None:
        self.lock = lock
        self.output = output
        self.camera = VisionSensor(f'/youBot/Vision_sensor', Angle(60, True), (640, 640), 1e-4, 10.)
        self.model = YOLO('yolo-seg-v3.pt')
        self.threshold = 0.5
        self.scan()
        
        
    def scan(self) -> None:
        BRICK_DIMS = np.array([.05500, .11458, .02979])
        
        while True:
            # start = time()
            matrix, depth, img = self.camera.shoot(do_rgb=True)
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f'dataset/{i}.png', frame)
            # i += 1
            result = self.model.predict(frame, verbose=False)[0]
            for c in result:
                conf = round(c.boxes.conf[0].item(), 2)
                if conf < 0.5:
                    continue
                
                coords = [round(x) for x in c.boxes.xyxy.tolist()[0]]
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                c_mask = np.zeros(img.shape[:2], np.uint8)
                try:
                    cv2.drawContours(c_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                except cv2.error:
                    continue
                mask = c_mask[coords[1]:coords[3], coords[0]:coords[2]].astype(bool)
                try:
                    cloud = self.camera.get_cloud(depth, matrix, tuple(coords[:2]), tuple(coords[2:]))[:, mask.flatten()]
                    planes = find_planes(cloud)
                    planes = remove_non_orthogonal_planes(cloud, planes)
                    pos, basis = construct_basis(cloud, planes, BRICK_DIMS)
                    dist = np.linalg.norm(np.array(self.camera.get_pos()-pos))
                    rot = np.nan
                    if dist < 0.5:
                        rot = get_orientation(basis)[2]
                    self.output.put((pos, rot, dist))
                except Exception as ex:
                    pass
                    # print(f'VISION: {ex}')
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.imshow('camera', frame)
            cv2.waitKey(1)
            # print(f'FPS: {round(1/(time()-start))}')
            
    
    
class Lidar:
    
    def __init__(self, lock, output: Queue, scale: float = 0.1) -> None:
        self.scale = scale
        self.lock = lock
        self.output = output
        self.lidar = VisionSensor(f'/youBot/Lidar', Angle(135, True), (1, 680), 1e-4, 10.)
        self.scan()
        
            
    def scan(self) -> None:
        while True:
            matrix, depth = self.lidar.shoot()
            try:
                cloud = self.lidar.get_cloud(depth, matrix)[:2].transpose()
                measurement = cloud2map(cloud[~np.isnan(cloud).any(axis=1)], self.scale)
                self.output.put(measurement)
            except Exception as ex:
                    print(f'LIDAR: {ex}')
            sleep(0.1)


class State(Enum):
    
    IDLE = 0
    COLLECTING = 1
    PICKING = 2
    LOADED = 3
    DELIVERING = 4
    BUILDING = 5
    FINISHED = 6
     
            
class Control:
    
    def __init__(self, lock, output: Queue) -> None:
        self.lock = lock
        self.output = output
        self.alive = True
        self.scale = 0.1
        self.robot = assemble_robot()
        self.state: State = State.IDLE
        self.path = []
        self.is_path_free = True
        self.scheme = open_scheme('schemes/pyramid_scheme.json')
        self.progress = self.scheme.iterator()
        self.navigator = Navigator(Map((round(-10/self.scale), round(-10/self.scale)), (round(20/self.scale), round(20/self.scale)), self.scale))
        self.cluster_system = ClusterSystem()
        self.cluster_system.TOLERANCE = .3
        c_thr = Thread(target=self.main)
        u_thr = Thread(target=self.sensor_updater, daemon=True)
        u_thr.start()
        c_thr.start()
        c_thr.join()
        u_thr.join()
    
    
    def make_route(self, destination: np.ndarray | tuple):
        trajectory = self.navigator.go_to(np.array(self.robot.platform.ref.get_xy()), np.array(destination), 'rrt')
        self.path = list(np.vstack([np.linspace(trajectory[i], point, 5, endpoint=False) for i, point in enumerate(trajectory[1:])]))
        self.is_path_free = True
    
    
    def check_path(self) -> bool:
        rrt = RRT_star(np.array([]), np.array([]), self.navigator.map)
        return all([rrt.check_intersection(self.path[i], point, 20) for i, point in enumerate(self.path[1:])])
    
    
    def choose_building_point(self, target_pos: np.ndarray, target_rot: float, access_check: bool = True) -> np.ndarray:
        vector = np.array([1.2*self.robot.arm.max_grab_distance, 0.])
        building_points = np.array([rotate2D(tuple(target_pos), target_pos+vector, target_rot-angle) for angle in (PI/2, -PI/2)]) # type: ignore
        bitmap = self.navigator.map.get_bitmap()
        if access_check:
            accessable = np.array([bitmap[*self.navigator.map.absolute2map(point).astype(np.int16)] for point in building_points])
        else:
            accessable = np.full(2, fill_value=1).astype(bool)
        robot_pos = self.robot.platform.ref.get_xy()
        try:
            return min(building_points[accessable], key=lambda p: np.linalg.norm(p-robot_pos))
        except:
            raise RuntimeError('Unreachable building task')
            
    
    def get_target(self):
        robot_pos = np.array(self.robot.platform.ref.get_xy())
        cluster = self.cluster_system.get_nearest(robot_pos)
        return cluster
    

    def go_to(self, destination):
        self.robot.platform.stop()
        print(f'Making route to {destination}...')
        dist = norm(np.array(destination)-np.array(self.robot.platform.ref.get_xy()))
        if dist < .5:
            self.robot.platform.travel_to(tuple(destination))
            return 
        start = time()
        self.make_route(destination)
        print(f'{time()-start:.3f} s {norm(np.array(destination)-np.array(self.robot.platform.ref.get_xy())):.3f} m')
        while True:
            if not self.is_path_free:
                self.robot.platform.stop()
                print('Route obstructed')
                print(f'Making route to {destination}...')
                start = time()
                self.make_route(destination)
                print(f'{time()-start:.3f} s {norm(np.array(destination)-np.array(self.robot.platform.ref.get_xy())):.3f} m')
            elif len(self.path) == 0:
                break
            else:
                self.robot.platform.travel_to(tuple(self.path.pop()))
            
    
    
    def main(self):
        Sim.lock = self.lock
        end = time() + np.inf
        self.robot.arm.park(False)
        self.robot.platform.stop()
        sleep(1)
        while time() < end:
            match self.state:
                case State.IDLE:
                    print(self.state)
                    try:
                        target_cluster = self.get_target()
                        self.state = State.COLLECTING
                    except RuntimeError:
                        self.robot.platform.rotate_to(Angle(0))
                        self.robot.platform.rotate_to(Angle(2*PI/3))
                        self.robot.platform.rotate_to(Angle(-2*PI/3))
                        self.robot.platform.rotate_to(Angle(0))
                    
                case State.COLLECTING:
                    print(self.state)
                    print(target_cluster.center[:2])
                    target_pos = target_cluster.center[:2]
                    robot_pos = np.array(self.robot.platform.ref.get_xy())
                    approach_pos = self.navigator.map.find_closest_to(target_pos, robot_pos)
                    approach_pos += .1*(approach_pos - target_pos)
                    try:
                        self.go_to(tuple(approach_pos))
                    except IndexError:
                        continue
                    except RuntimeError:
                        self.robot.platform.get_back()
                        self.state = State.IDLE
                        continue
                    self.state = State.PICKING
                    
                case State.PICKING:
                    print(self.state)
                    self.robot.platform.rotate_to(point=tuple(target_cluster.center[:2]))
                    sleep(2)
                    self.robot.approach(tuple(target_cluster.center[:2]))
                    sleep(2)
                    if np.isnan(target_cluster.rot):
                        self.robot.platform.get_back()
                        self.robot.platform.rotate_to(point=tuple(target_cluster.center[:2]))
                        sleep(4)
                    if np.isnan(target_cluster.rot):
                        print('False detections removed')
                        self.cluster_system.drop_clusters(target_cluster.center, 0.5)
                        self.state = State.IDLE
                        continue
                    try:
                        self.robot.arm.take(tuple(target_cluster.center), Angle(PI/2-target_cluster.rot))
                    except Warning:
                        self.robot.platform.get_back()
                        self.cluster_system.drop_clusters(target_cluster.center, 0.5)
                        self.state = State.IDLE
                        continue
                    self.robot.arm.park()
                    self.cluster_system.drop_clusters(target_cluster.center, 0.5)
                    self.navigator.map.cut(target_cluster.center[:2], 0.5)
                    self.state = State.LOADED
                    
                case State.LOADED:
                    print(self.state)
                    try:
                        putting_point, rot, _ = next(self.progress)
                    except StopIteration:
                        self.state = State.FINISHED
                        continue
                    self.state = State.DELIVERING
                    
                case State.DELIVERING:
                    print(self.state)
                    try:
                        approach_pos = self.choose_building_point(np.array(putting_point[:2]), rot)
                    except RuntimeError as err:
                        print(err)
                        print('Trying to get as close as possible')
                        robot_pos = np.array(self.robot.platform.ref.get_xy())
                        approach_pos = self.navigator.map.find_closest_to(self.choose_building_point(np.array(putting_point[:2]), rot, False), robot_pos)
                    try:
                        self.go_to(tuple(approach_pos))
                    except IndexError:
                        continue
                    except RuntimeError:
                        self.robot.platform.get_back()
                        continue
                    
                    self.robot.approach(putting_point[:2])
                    self.state = State.BUILDING
                    
                case State.BUILDING:
                    print(self.state)
                    self.cluster_system.restrict(putting_point, 1.)
                    try:
                        self.robot.arm.put(putting_point, Angle(rot))
                    except Warning:
                        self.state = State.DELIVERING
                        continue
                    self.robot.arm.park()
                    self.robot.platform.get_back()
                    self.robot.platform.get_back()
                    self.cluster_system.drop_clusters(np.array(putting_point), 0.5)
                    sleep(1)
                    self.state = State.IDLE
                
                case State.FINISHED:
                    print(self.state)
                    break
                
                case _:
                    raise ValueError(f'Unknown state: {self.state}')
        
        self.robot.platform.stop()
        self.robot.arm.park()
        self.alive = False
        cv2.destroyAllWindows()
    
    
    def sensor_updater(self):
        with self.lock:
            drawing_obj = Sim.sim.addDrawingObject(Sim.sim.drawing_spherepts, 0.005, 0, -1, 0, [0, 0, 1])
        with open('detection_points.log', 'w') as f:
            while self.alive:
                if self.output.empty():
                    continue
                
                content = self.output.get()
                if isinstance(content, Map):
                    self.navigator.map.update(content)
                    rgb = cv2.cvtColor(self.navigator.map.array.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    points = [self.navigator.map.absolute2map(center[:2]).astype(int) for center in self.cluster_system.centers]
                    for point in points:
                        rgb[tuple(point)] = np.array([0, 0, 255])
                    rgb[tuple(self.navigator.map.absolute2map(np.array(self.robot.platform.ref.get_xy())).astype(int))] = np.array([0, 255, 0])
                    scaled_up = cv2.resize(rgb, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)

                    cv2.imshow('map', scaled_up)
                    cv2.waitKey(1)
                    if len(self.path) > 0:
                        self.is_path_free = self.check_path()
                    # print('map updated')
                elif isinstance(content, tuple):
                    pos, rot, dist = content
                    if self.state in (State.LOADED, State.DELIVERING, State.BUILDING):
                        continue
                    self.cluster_system.add_item(Detection(*content))
                    with self.lock:
                        Sim.sim.removeDrawingObject(drawing_obj)
                        drawing_obj = Sim.sim.addDrawingObject(Sim.sim.drawing_spherepts, 0.005, 0, -1, 0, [0, 0, 1])
                    brick = icp.get_origin(0.0550, 0.1146, 0.0298)
                    detected_brick = icp.rotate(brick, rot) + pos
                    with self.lock:
                        for point in detected_brick: Sim.sim.addDrawingObjectItem(drawing_obj, list(point))
                    # print(content)
                    f.write(f'{pos[0]} {pos[1]} {pos[2]} {rot}\n')


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    Simulation().start()