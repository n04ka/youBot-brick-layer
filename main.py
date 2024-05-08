from kuka import *
from time import time
from navigation import *
import icp
from ultralytics import YOLO
from itertools import cycle



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
        self.model = YOLO('best.pt')
        self.threshold = 0.6
        self.scan()
        
        
    def scan(self) -> None:
        # i = 2869
        while True:
            # start = time()
            matrix, depth, img = self.camera.shoot(do_rgb=True)
            frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f'dataset/{i}.png', frame)
            # i += 1
            # result = self.model.predict(frame, verbose=False)[0]
            # for box in result.boxes:
            #     conf = round(box.conf[0].item(), 2)
            #     if conf < self.threshold:
            #         continue
            #     coords = [round(x) for x in box.xyxy[0].tolist()]
            #     # print("Coordinates:", coords)
            #     # print("Probability:", conf)
            #     cut_rgb = img[coords[1]:coords[3], coords[0]:coords[2]]
            #     mask = np.array([[pixel[2] < pixel[0] + pixel[1] for pixel in row] for row in cut_rgb], dtype=bool).flatten()
            #     cloud = self.camera.get_cloud(depth, matrix, tuple(coords[:2]), tuple(coords[2:]))[:, mask]
            #     if cloud.shape[1] < 10: continue
            #     try:
            #         corners = icp.get_corners(cloud)
            #     except ValueError:
            #         continue
            #     pos, rot = icp.icp(corners, icp.get_origin(0.0550, 0.1146, 0.0298), 4)
            #     self.output.put((pos, rot))
            #     cv2.rectangle(frame, coords[:2], coords[2:], (0, 255, 0))
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
            cloud = self.lidar.get_cloud(depth, matrix)[:2].transpose()
            measurement = cloud2map(cloud[~np.isnan(cloud).any(axis=1)], self.scale)
            self.output.put(measurement)
            

class Control:
    
    def __init__(self, lock, output: Queue) -> None:
        self.lock = lock
        self.output = output
        self.alive = True
        self.scale = 0.1
        self.robot = assemble_robot()
        self.navigator = Navigator(Map((0, 0), (0, 0), self.scale))
        c_thr = Thread(target=self.main)
        u_thr = Thread(target=self.sensor_updater, daemon=True)
        u_thr.start()
        c_thr.start()
        c_thr.join()
        u_thr.join()
        
        
    def main(self):
        Sim.lock = self.lock
        end = time() + 60
        # self.robot.arm.park(False)
        self.robot.platform.stop()
        # angles = cycle((Angle(0), Angle(PI/2), Angle(PI), Angle(-PI/2)))
        dest = cycle(((1.2, -1.2), (1.2, -0.8), (0.8, -0.8), (0.8, -1.2)))
        while time() < end:
            self.robot.platform.move_to(next(dest))
            self.robot.platform.rotate_to(Angle(np.random.random()*2*PI))
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
                    # print('map updated')
                elif isinstance(content, tuple):
                    with self.lock:
                        Sim.sim.removeDrawingObject(drawing_obj)
                        drawing_obj = Sim.sim.addDrawingObject(Sim.sim.drawing_spherepts, 0.005, 0, -1, 0, [0, 0, 1])
                    pos, rot = content
                    
                    brick = icp.get_origin(0.0550, 0.1146, 0.0298)
                    detected_brick = icp.rotate(brick, rot) + pos
                    with self.lock:
                        for point in detected_brick: Sim.sim.addDrawingObjectItem(drawing_obj, list(point))
                    # print(content)
                    f.write(f'{pos[0]} {pos[1]} {pos[2]} {rot}\n')


if __name__ == '__main__':
    Simulation().start()