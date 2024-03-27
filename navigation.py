import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
from threading import Thread
import networkx as nx
from scipy.spatial import Voronoi
import cv2 as cv2


pg.init()
SCALE = 0.1
SCREEN = pg.display.set_mode((0, 0), pg.FULLSCREEN)
CLOCK = pg.time.Clock()


class Map:
    
    def __init__(self, pos: tuple, shape: tuple, scale: float) -> None:
        self.pos = np.array(pos, dtype=np.int16)
        self.array = np.zeros(shape, dtype=np.uint16)
        self.scale = scale


    def absolute2map(self, coords: np.ndarray) -> tuple[int, int]:
        return tuple(((coords - self.pos) // self.scale).astype(int))
    

    def map2absolute(self, coords: tuple | np.ndarray) -> np.ndarray:
        return np.array(coords) * self.scale + self.pos

    
    def __contains__(self, value) -> bool:
        i = np.array(value)
        low, high = self.limits()
        return all(i >= low) and all(i < high)
    
    
    def __getitem__(self, key) -> int | np.ndarray:
        if key not in self:
            raise IndexError(f'За границами карты {key}')
        i = np.array(key)
        index = i - self.pos[:i.size]
        return self.array[*index]
    
    
    def __setitem__(self, key, value: int):
        if key not in self:
            raise IndexError(f'За границами карты {key}')
        i = np.array(key)
        index = i - self.pos[:i.size]
        self.array[*index] = value


    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pos, self.pos + self.array.shape


    def update(self, other):
        if self.scale != other.scale:
            raise ValueError(f'Масштаб карт при сложении отличается: {self.scale} и {other.scale}')
        low1, high1 = self.limits()
        low2, high2 = other.limits()
        new_pos = np.min(np.vstack((low1, low2)), axis=0)
        new_high = np.max(np.vstack((high1, high2)), axis=0)
        new_array = np.zeros(new_high-new_pos, dtype=np.uint16)
        new_array[*[slice(l, h) for l, h in zip(low1-new_pos, low1-new_pos + np.array(self.array.shape))]] += self.array
        new_array[*[slice(l, h) for l, h in zip(low2-new_pos, low2-new_pos + np.array(other.array.shape))]] += other.array
        self.pos = new_pos
        self.array = new_array
        

    def show(self):
        x_min, y_max, x_max, y_min = np.concatenate(self.limits())*self.scale
        plt.imshow(self.array, cmap='gray_r', extent=[y_max, y_min, x_max, x_min])
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.grid(which='both')


def new_measurement(coords: np.ndarray, data: np.ndarray, scale: float) -> Map:
    x, y, rot = coords
    angle = np.linspace(-2*np.pi/3, 2*np.pi/3, len(data))
    data[data == 5.6] = np.nan
    data[data < 0.3] = np.nan
    xs = np.array([x + data * np.cos(rot-angle)]).transpose()
    ys = np.array([y + data * np.sin(rot-angle)]).transpose()
    points = np.hstack((xs, ys))
    points = points[~np.isnan(points).any(axis=1)]
    decomposed = (points // np.array([scale, scale])).astype(np.int16)
    decoposed_coords, counts = np.unique(decomposed, axis=0, return_counts=True)
    x_min, y_min, x_max, y_max = *decoposed_coords.min(axis=0), *decoposed_coords.max(axis=0)
    size = np.array([x_max - x_min + 1, y_max - y_min + 1])
    TOPLEFT = np.array([x_min, y_min])
    cell_map = np.zeros(size, dtype=np.uint16)
    for coord, n in zip(decoposed_coords, counts):
        cell_map[*(coord-TOPLEFT)] = n
    cell_map = cell_map.transpose()
    obj = Map(tuple(reversed(TOPLEFT)), cell_map.shape, scale)
    obj.array = cell_map
    return obj


def gray(im):
    im = 255 * (1 - im / 255)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


class Robot(pg.sprite.Sprite):
    
    def __init__(self) -> None:
        super().__init__()
        self.rect = pg.Rect(0, 0, 0.531//SCALE*10, 0.380//SCALE*10)
        self.rot = 0.
        self.update()
    
    
    def move(self, coords: np.ndarray):
        self.rect.center = tuple(coords[:2]) #type: ignore
        self.rot = -coords[2]
     
    
    def update(self):
        center = self.rect.center #type: ignore
        surf = pg.Surface((0.531//SCALE*10, 0.380//SCALE*10), pg.SRCALPHA) #type: ignore
        surf.fill((255, 0, 0))
        self.image = pg.transform.rotate(surf, self.rot/np.pi*180)
        self.rect = self.image.get_rect()
        self.rect.center = center


class Navigator:

    def __init__(self, world_map: Map, robot: Robot) -> None:
        self.map = world_map
        self.robot = robot
        self.graph = nx.Graph()
        self.methods = {'cell' : self.cell_graph,
                        'voronoi' : self.voronoi_graph,
                        'A*' : self. a_star}


    def go_to(self, target: np.ndarray, algorithm: str = 'A*', graph_creation: str = 'cell') -> list[tuple]:
        bitmap = ~cv2.threshold(self.map.array, 10, 1, cv2.THRESH_BINARY)[1].astype(bool)
        plt.subplot(1, 2, 1)
        plt.gca().set_aspect('equal')
        plt.imshow(bitmap)
        self.graph = self.methods[graph_creation](bitmap)
        path = self.methods[algorithm](self.graph, target)
        return path


    def cell_graph(self, bitmap: np.ndarray) -> nx.Graph:
        xs, ys = np.array(range(bitmap.shape[0])), np.array(range(bitmap.shape[1]))
        cells = np.stack(np.meshgrid(xs, ys), axis=2)
        free_cells = cells[bitmap.transpose()].reshape((-1, 2))

        graph = nx.Graph()
        graph.add_nodes_from([tuple(coord) for coord in free_cells])

        for node in graph.nodes:
            coord = np.array(node)
            for move in ((0, 1), (1, 0), (1, 1), (1, -1)):
                neighbour = tuple(coord + np.array(move))
                if neighbour not in graph.nodes:
                    continue
                checks = {(1, 1) : [coord+np.array([1, 0]), coord+np.array([0, 1])],
                        (1, -1) : [coord+np.array([0, -1]), coord+np.array([1, 0])]}
                if move in checks:
                    if not all([tuple(block) in graph.nodes for block in checks[move]]):
                        continue
                graph.add_edge(node, neighbour, weight=np.linalg.norm(np.array(move)))
        plt.subplot(1, 2, 2)
        plt.gca().invert_yaxis()
        plt.gca().set_aspect('equal')
        nx.draw(graph, pos={node : node[::-1] for node in graph.nodes}, node_size=2, width=0.5)
        plt.show()
        return graph


    def voronoi_graph(self, bitmap: np.ndarray) -> nx.Graph:
        xs, ys = np.array(range(bitmap.shape[0])), np.array(range(bitmap.shape[1])), 
        cells = np.stack(np.meshgrid(xs, ys), axis=2)
        obstacles = cells[~bitmap.transpose()].reshape((-1, 2))

        diagram = Voronoi(obstacles)
        pos = {i : coord for i, coord in enumerate(diagram.vertices)}
        graph = nx.Graph()
        graph.add_nodes_from(pos)
        graph.add_node(-1)
        graph.add_edges_from(diagram.ridge_vertices)
        graph.remove_node(-1)

        for node in pos:
            try:
                if not bitmap[*np.round(pos[node]).astype(int)]:
                    graph.remove_node(node)
            except IndexError:
                graph.remove_node(node)
        
        nx.set_edge_attributes(graph, {(start, end) : np.linalg.norm(pos[start]-pos[end]) for start, end in graph.edges}, 'weight')
        return graph


    def a_star(self, graph: nx.graph, target: tuple) -> list:
        norm = np.linalg.norm
        START = self.map.absolute2map(display.display2absolute(np.array(self.robot.rect.center)))
        FINISH = target
        nx.set_node_attributes(graph, {node : norm(np.array(FINISH)-np.array(node)) for node in graph.nodes}, 'weight')
        nx.set_node_attributes(graph, np.inf, 'path')
        nx.set_node_attributes(graph, {START : 0.}, 'path')
        nx.set_node_attributes(graph, {START : graph.nodes[START]['weight']}, 'cost')

        front: dict[tuple, tuple | None] = {START : None}
        visited = {}

        while True:
            new_node = min(front, key = lambda node: graph.nodes[node]['cost'])
            visited[new_node] = front[new_node]
            front.pop(new_node)
            
            if new_node == FINISH:
                break
            
            for n in graph.neighbors(new_node):
                if n in visited:
                    continue
                if graph.nodes[new_node]['path'] + graph.edges[n, new_node]['weight'] < graph.nodes[n]['path'] or n not in front:
                    nx.set_node_attributes(graph, {n : graph.nodes[new_node]['path'] + graph.edges[n, new_node]['weight']}, 'path')
                    nx.set_node_attributes(graph, {n : graph.nodes[n]['path'] + graph.nodes[n]['weight']}, 'cost')
                    front[n] = new_node

        path = [FINISH]
        while path[-1]:
            path.append(visited[path[-1]])
        return path


class MapDisplay(pg.sprite.Sprite):
    
    def __init__(self, owner: Map, scale: float = 10.) -> None:
        super().__init__()
        self.owner = owner
        self.scale = scale
        self.rect = pg.Rect(200, 0, SCREEN.get_width()-200, SCREEN.get_height())
        self.update()
        
    
    def absolute2display(self, coords: np.ndarray):
        return (coords / self.owner.scale - self.owner.pos[::-1]) * self.scale
    
    
    def display2absolute(self, coords: np.ndarray):
        return (coords / self.scale + self.owner.pos[::-1]) * self.owner.scale
    

    def draw_path(self, path: list[tuple]):
        pg.draw.aalines(self.image, (0, 0, 255), True, path)

    
    def update(self):
        self.image = pg.transform.scale_by(pg.surfarray.make_surface(gray(self.owner.array.transpose())), self.scale)


world_map = Map((0, 0), (0, 0), SCALE)
robot = Robot()
display = MapDisplay(world_map)
navigator = Navigator(world_map, robot)


def update_map():
    with open(f"examples/examp7.txt") as file:
        for line in file:
            c, d = line.split("; ")
            coords = np.array([float(number) for number in c.split(", ")])
            data = np.array([float(number) for number in d.split(", ")])
            world_map.update(new_measurement(coords, data, SCALE))
            robot.move(np.array([*display.absolute2display(coords[:2]), coords[2]]))
            CLOCK.tick(20)
            
            
def visualize():
    running = True  
    while running:

        displays.update()
        robots.update()

        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = event.pos
                target = display.display2absolute(np.array([x-display.rect.topleft[0], y]))
                path = navigator.go_to(target)
                display.draw_path(path)
                print(target)
                      
        SCREEN.fill((128, 128, 128))
        robots.draw(display.image)
        displays.draw(SCREEN)
        font = pg.font.SysFont(None, 24)
        x_r, y_r = display.display2absolute(np.array(robot.rect.center))
        x_m, y_m = pg.mouse.get_pos()
        lines = [f'Robot pos: {x_r:.2f}, {y_r:.2f}',
                 f'Screen mouse pos: {x_m}, {y_m}',
                 f'Display mouse pos: {x_m-200}, {y_m}',
                 f'Absolute mouse pos: {display.display2absolute(np.array([x_m-200, y_m]))}',
                 f'Map mouse pos: {world_map.absolute2map(display.display2absolute(np.array([x_m-200, y_m])))}']
        for i, line in enumerate(lines):
            surf = font.render(line, True, (0, 0, 0))
            SCREEN.blit(surf, (20, i*20+20))
        CLOCK.tick(60)
        pg.display.flip()
    pg.quit()


displays: pg.sprite.GroupSingle = pg.sprite.GroupSingle(display)
robots: pg.sprite.GroupSingle = pg.sprite.GroupSingle(robot)
update_thread = Thread(target=update_map, daemon=True)
update_thread.start()
visualize()
update_thread.join()