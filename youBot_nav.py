from navigation import *
from kuka import *


class Control:
    
    def __init__(self, scale: float = 0.1, display_size: int = 4) -> None:
                
        Sim.sim.startSimulation()
        pg.init()
        self.screen = pg.display.set_mode((800, 800))
        self.scale = scale
        self.robot = Robot(0.1, display_size)
        self.youBot = assemble_robot()
        self.youBot.arm.park(False)
        self.map = Map((round(-10/scale), round(-10/scale)), (round(20/scale), round(20/scale)), scale)
        self.navigator = Navigator(self.map)
        self.display = MapDisplay(self.map, self.screen, self.robot, display_size)
        self.path = []


    def go_to(self, target: np.ndarray):
        start = np.array(self.youBot.platform.ref.get_xy())
        try:
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
        else:
            print('Path obstructed. Refinding...')
            self.go_to(self.path[0])
    
    
    def visualize(self):
        clock = pg.time.Clock()
        running = True
        robot: pg.sprite.GroupSingle = pg.sprite.GroupSingle(self.robot)
        display : pg.sprite.GroupSingle = pg.sprite.GroupSingle(self.display)
        while running:
            self.follow_path()
            cloud = self.youBot.lidar.get_cloud()
            self.map.update(cloud2map(cloud[:2].transpose(), self.scale))
            
            x, y = self.youBot.platform.ref.get_xy()
            pos = self.display.absolute2display(np.array([x, y]))
            self.robot.move(np.array([*pos, self.youBot.platform.ref.get_azimuth()]))
            robot.update()
            display.update()

            for event in pg.event.get():
                if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                    running = False
                elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    print('Finding path...')
                    self.go_to(self.display.display2absolute(np.array(event.pos)))
                        
            self.screen.fill((128, 128, 128))
            if self.path:
                self.display.draw_path(self.path + [np.array([x, y])])
            display.draw(self.screen)
            robot.draw(self.screen)
            clock.tick(15)
            pg.display.flip()
        pg.quit()
        Sim.sim.stopSimulation()


if __name__ == '__main__':
    Control().visualize()