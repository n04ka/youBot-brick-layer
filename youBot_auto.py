from navigation import *
from kuka import *
from schematics import *


class Control:
    
    def __init__(self, scale: float = 0.1) -> None:
                
        Sim.sim.startSimulation()
        sleep(0.1)
        self.scale = scale
        self.youBot = assemble_robot()
        self.youBot.arm.park(False)
        self.map = Map((round(-10/scale), round(-10/scale)), (round(20/scale), round(20/scale)), scale)
        self.navigator = Navigator(self.map)
        self.path = []
        self.scheme = open_scheme('Navigation\\schemes\\default_scheme.json')


    def go_to(self, target: np.ndarray):
        start = np.array(self.youBot.platform.ref.get_xy())
        try:
            print('Looking for path...')
            trajectory = self.navigator.go_to(start, target, 'rrt')
            self.path = list(np.vstack([np.linspace(trajectory[i], point, 5, endpoint=False) for i, point in enumerate(trajectory[1:])]))
        except:
            print('Pathfinding error')
        
    
    def follow_path(self):
        if not self.path:
            return
        
        rrt = RRT_star(np.array([]), np.array([]), self.map)
        if all([rrt.check_intersection(self.path[i], point, 20) for i, point in enumerate(self.path[1:])]):
            self.youBot.platform.travel_to(tuple(self.path[-1]))
            self.path.pop()
            if not self.path:
                print('Destination reached!')
                if self.state == 'collecting':
                    self.state = 'picking'
                elif self.state == 'delivering':
                    self.state = 'building'
        else:
            print('Path obstructed.')
            if self.state == 'collecting':
                self.go_to(self.map.find_closest_to(np.array(self.brick.get_xy())))
            elif self.state == 'delivering':
                self.go_to(self.map.find_closest_to(np.array(self.task.get_pos()[:2])))
    
    
    def main(self):
        
        self.scheme.show()
        plt.show()
        bricks = [Brick(f'/Brick{i+1}') for i in range(10)]
        tasks = [Task(Pose(*coords, gamma=rot)) for coords, rot, _ in self.scheme.iterator()]
        disp = Dispatcher(self.youBot, bricks, tasks)
        
        self.state = 'ready'
        
        
        while Sim.sim.getSimulationState():
            cloud = self.youBot.lidar.get_cloud()
            self.map.update(cloud2map(cloud[:2].transpose(), self.scale))
            
            if len(disp._materials) == 0 or len(disp._tasks) == 0:
                print('Work done')
                break
            print(self.state)
            match self.state:
                case 'ready':
                    self.brick = disp.reserve_material()
                    self.go_to(self.map.find_closest_to(np.array(self.brick.get_xy())))
                    self.state = 'collecting'
                case 'collecting':
                    self.follow_path()
                case 'picking':
                    self.youBot.go_pick(self.brick)
                    self.state = 'loaded'
                case 'loaded':
                    self.task = disp.choose_task()
                    self.go_to(self.map.find_closest_to(np.array(self.task.get_pos()[:2])))
                    self.state = 'delivering'
                case 'delivering':
                    print(self.path[0])
                    self.follow_path()
                case 'building':
                    self.youBot.go_place(self.task)
                    self.state = 'ready'
                

        Sim.sim.stopSimulation()


if __name__ == '__main__':
    Control(0.1).main()