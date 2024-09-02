import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import pygame as pg
import networkx as nx
import cv2 as cv2


class Map:
    
    def __init__(self, pos: tuple, shape: tuple, scale: float) -> None:
        self.pos = np.array(pos, dtype=np.int16)
        self.array = np.zeros(shape, dtype=np.uint32)
        self.scale = scale


    def absolute2map(self, coords: np.ndarray) -> np.ndarray:
        x, y = (coords / self.scale - self.pos)[::-1]
        return np.array([self.array.shape[0]-x, y])

    def map2absolute(self, coords: np.ndarray) -> np.ndarray:
        x, y = coords
        return (np.array([self.array.shape[0]-x, y])[::-1] + self.pos) * self.scale

    
    def cut(self, center: np.ndarray, radius: float):
        c = self.absolute2map(center).astype(int)
        r = int(radius // self.scale)
        array = self.array.astype(np.float32)
        cv2.circle(array, c, r, 0, -1) # type: ignore
        self.array = array.astype(np.uint32)
    
    
    def get_bitmap(self) -> np.ndarray:
        bitmap = cv2.threshold(self.array.astype(np.float32), 10, 1, cv2.THRESH_BINARY)[1]
        kernel = np.zeros([int(np.ceil((0.54**2 + 0.4**2)**.5/self.scale))]*2, int)
        cv2.circle(kernel, np.array(kernel.shape)//2, kernel.shape[0]//2, 1, -1) # type: ignore
        return ~cv2.filter2D(bitmap, -1, kernel).astype(bool)
    
    
    def find_closest_to(self, target: np.ndarray, p_from: np.ndarray) -> np.ndarray:
        t = self.absolute2map(target)
        p_f = self.absolute2map(p_from)
        bitmap = self.get_bitmap().transpose()
        reachable_points = np.stack(np.meshgrid(np.arange(bitmap.shape[1]), np.arange(bitmap.shape[0])), axis=-1)[bitmap].reshape(-1, 2)
        if len(reachable_points) == 0:
            raise ValueError('No reachable points')
        closest = min(reachable_points, key=lambda p: np.linalg.norm(t-p) + np.linalg.norm(p_f-p))
        return self.map2absolute(closest)
    
    
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
        new_array = np.zeros(new_high-new_pos, dtype=np.uint32)
        new_array[*[slice(l, h) for l, h in zip(low1-new_pos, low1-new_pos + np.array(self.array.shape))]] += self.array
        new_array[*[slice(l, h) for l, h in zip(low2-new_pos, low2-new_pos + np.array(other.array.shape))]] += other.array
        self.pos = new_pos
        self.array = new_array
        

    def show(self):
        x_min, y_max, x_max, y_min = np.concatenate(self.limits())*self.scale
        plt.imshow(self.array, cmap='gray_r', vmax=10, extent=[y_max, y_min, x_min, x_max])
        plt.gca().set_aspect('equal')
        plt.colorbar()
        plt.grid(which='both')


class RRT_star:
    
    def __init__(self, start: np.ndarray, finish: np.ndarray, world_map: Map, segment: float = 0.3) -> None:
        self.graph = nx.DiGraph()
        self.pos = dict()
        self.map = world_map
        self.bitmap = world_map.get_bitmap()
        self.start = start
        self.finish = finish
        self.segment = segment
        self.pos[0] = start
        self.graph.add_node(0, weight=0.)
        self.finish_node = None
    
    
    def check_intersection(self, start: np.ndarray, end: np.ndarray, num: int = 20) -> bool:
        line = np.linspace(start, end, num, endpoint=True)
        return all([self.bitmap[*self.map.absolute2map(point).astype(np.int16)] for point in line])
    
        
    def recalculate_weights(self, root: int):
        for node in nx.descendants(self.graph, root):
            pred = next(self.graph.predecessors(node))
            self.graph.nodes[node]['weight'] = self.graph.nodes[pred]['weight'] + norm(self.pos[pred] - self.pos[node])


    def update_graph(self, point: np.ndarray) -> int | None:
        
        closest = min(self.graph, key = lambda node: norm(self.pos[node]-point))
        projection = self.segment*(point-self.pos[closest])/norm(point-self.pos[closest])
        new_point = point if norm(point-self.pos[closest]) < norm(projection) else self.pos[closest] + projection

        neighbours = {node : self.graph.nodes[node]['weight'] + norm(self.pos[node]-new_point)
                    for node in self.pos
                    if norm(self.pos[node]-new_point) <= self.segment*3 and 
                    self.check_intersection(self.pos[node], new_point)}
    
        if not neighbours:
            return
        
        # if self.graph.number_of_nodes() > 1000:
        #     self.map.show()
        #     nx.draw(self.graph, self.pos, node_size=5, width=1)
        #     plt.axis('on')
        #     plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        #     plt.show()    
        #     raise RuntimeError('Cannot find the path') 
        
        node = len(self.pos)
        self.pos[node] = new_point
        best_neighbour = min(neighbours, key=neighbours.get) # type: ignore
        self.graph.add_node(node, weight=self.graph.nodes[best_neighbour]['weight'] + norm(self.pos[best_neighbour]-new_point))
        self.graph.add_edge(best_neighbour, node)
        
        for neighbour in neighbours:
            if self.graph.nodes[neighbour]['weight'] > self.graph.nodes[node]['weight'] + norm(self.pos[neighbour]-new_point):
                self.graph.remove_edge(list(self.graph.predecessors(neighbour))[0], neighbour)
                self.graph.add_edge(node, neighbour)
                
        self.recalculate_weights(node)
        return node
    
    
    def run(self):
        if not self.bitmap[*self.map.absolute2map(self.finish).astype(np.int16)]:
            raise RuntimeError('Finish in obstacle')
        if not self.bitmap[*self.map.absolute2map(self.start).astype(np.int16)]:
            raise RuntimeError('Start in obstacle')
        i = 0
        searching = True
        iterations = 0
        while searching or i < 200:
            if iterations > 2000 and searching:
                raise IndexError('Struggling to find route')
            iterations += 1
            new_point = np.random.uniform(*np.array(self.map.limits())[::-1]*self.map.scale)
            added_node = self.update_graph(new_point)
            
            if not added_node:
                continue
            
            dist_to_finish = norm(self.pos[added_node] - self.finish)
            if dist_to_finish < self.segment and self.check_intersection(self.pos[added_node], self.finish):
                node = len(self.pos)
                self.pos[node] = self.finish
                finish_node = node
                self.graph.add_node(node, weight=self.graph.nodes[added_node]['weight'] + dist_to_finish)
                self.graph.add_edge(added_node, node)
                if searching:
                    print('path found')
                searching = False
                
            
            if not searching:
                i += 1 

        
        # self.map.show()
        # nx.draw(self.graph, self.pos, node_size=5, width=1)
        # plt.axis('on')
        # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()     
        path = [finish_node]
        while path[-1]:
            path.append(*self.graph.predecessors(path[-1]))
        
        # plt.figure()
        # plt.imshow(self.bitmap)
        # nx.draw(self.graph, pos={node : (self.map.absolute2map(coords)-self.map.pos)[::-1] for node, coords in self.pos.items()}, node_size=5, width=1)
        # plt.axis('on')
        # plt.gca().tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()
        return [self.pos[node] for node in path]


def cloud2map(cloud: np.ndarray, scale: float) -> Map:
    cloud[:, 1] *= -1
    # plt.scatter(*cloud.transpose(), s=0.1)
    # plt.gca().set_aspect('equal')
    # plt.title('Облако в системе координат карты')
    # plt.show()
    decomposed = (cloud // scale).astype(int)
    decoposed_coords, counts = np.unique(decomposed, axis=0, return_counts=True)
    x_min, y_min, x_max, y_max = *decoposed_coords.min(axis=0), *decoposed_coords.max(axis=0)
    size = np.array([x_max - x_min + 1, y_max - y_min + 1])
    TOPLEFT = np.array([x_min, y_min])
    cell_map = np.zeros(size, dtype=np.uint32)
    for coord, n in zip(decoposed_coords, counts):
        cell_map[*(coord-TOPLEFT)] = n
    cell_map = cell_map.transpose()
    obj = Map(tuple(reversed(TOPLEFT)), cell_map.shape, scale)
    obj.array = cell_map
    return obj


def new_measurement(coords: np.ndarray, data: np.ndarray, scale: float) -> Map:
    x, y, rot = coords
    angle = np.linspace(-2*np.pi/3, 2*np.pi/3, len(data))
    data[data == 5.6] = np.nan
    data[data < 0.3] = np.nan
    xs = np.array([x + data * np.cos(rot-angle)]).transpose()
    ys = np.array([y + data * np.sin(rot-angle)]).transpose()
    points = np.hstack((xs, ys))
    points = points[~np.isnan(points).any(axis=1)]
    return cloud2map(points, scale)


def gray(im):
    im = 255 * (1 - im / 255)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret


class Robot(pg.sprite.Sprite):
    
    def __init__(self, scale: float, display_scale: float = 10) -> None:
        super().__init__()
        self.scale = scale
        self.display_scale = display_scale
        self.rect = pg.Rect(0, 0, 0.531//scale*self.display_scale, 0.380//scale*self.display_scale)
        self.rot = 0.
        self.update()
    
    
    def move(self, coords: np.ndarray):
        self.rect.center = tuple(coords[:2]) #type: ignore
        self.rot = coords[2]
     
    
    def update(self):
        center = self.rect.center #type: ignore
        surf = pg.Surface((0.531//self.scale*self.display_scale, 0.380//self.scale*self.display_scale), pg.SRCALPHA) #type: ignore
        surf.fill((255, 0, 0))
        self.image = pg.transform.rotate(surf, self.rot/np.pi*180)
        self.rect = self.image.get_rect()
        self.rect.center = center


class Navigator:

    def __init__(self, world_map: Map) -> None:
        self.map = world_map
        self.graph = nx.Graph()


    def go_to(self, start: np.ndarray, target: np.ndarray, algorithm: str = 'A*') -> list[np.ndarray]:
        if algorithm == 'A*':
            bitmap = self.map.get_bitmap()
            self.graph = self.cell_graph(bitmap)
            return self.a_star(self.graph, start, target)
        
        return self.rrt_star(start, target)


    def rrt_star(self, start: np.ndarray, target: np.ndarray) -> list:
        return RRT_star(start, target, self.map, .5).run()


    def cell_graph(self, bitmap: np.ndarray) -> nx.Graph:
        y_min, x_min, y_max, x_max = np.hstack(self.map.limits())
        xs = np.array(range(x_min, x_max))
        ys = np.array(range(y_min, y_max))
        cells = np.stack(np.meshgrid(ys, xs), axis=2)
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
        # plt.subplot(1, 2, 2)
        # plt.gca().invert_yaxis()
        # plt.gca().set_aspect('equal')
        # nx.draw(graph, pos={node : node[::-1] for node in graph.nodes}, node_size=2, width=0.5)
        # plt.show()
        return graph


    def a_star(self, graph: nx.Graph, start: np.ndarray, target: np.ndarray) -> list:
        START = tuple(self.map.absolute2map(start)) #type: ignore
        FINISH = tuple(self.map.absolute2map(target))
        print(f'START: {START}, FINISH: {FINISH}')
        assert START in graph.nodes
        if FINISH not in graph.nodes:
            print('Path does not exist')
            return [] 
        nx.set_node_attributes(graph, {node : norm(np.array(FINISH)-np.array(node)) for node in graph.nodes}, 'weight')
        nx.set_node_attributes(graph, np.inf, 'path')
        nx.set_node_attributes(graph, {START : 0.}, 'path')
        nx.set_node_attributes(graph, {START : graph.nodes[START]['weight']}, 'cost')

        front: dict[tuple, tuple | None] = {START : None}
        visited = {}

        while True:
            if not front:
                print('Path does not exist')
                return []
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
        return [self.map.map2absolute(np.array(node)) for node in path[:-1]]
      

class MapDisplay(pg.sprite.Sprite):
    
    def __init__(self, owner: Map, screen: pg.Surface, robot: Robot, scale: float = 10.) -> None:
        super().__init__()
        self.owner = owner
        self.robot = robot
        self.scale = scale
        self.rect = pg.Rect(0, 0, screen.get_width(), screen.get_height())
        self.path = []
        self.update()
        
    
    def absolute2display(self, coords: np.ndarray) -> np.ndarray:
        return self.map2display(self.owner.absolute2map(coords))
    
    
    def display2absolute(self, coords: np.ndarray) -> np.ndarray:
        return self.owner.map2absolute(self.display2map(coords))
    
    
    def map2display(self, coords: np.ndarray) -> np.ndarray:
        return coords[::-1] * self.scale
    
    
    def display2map(self, coords: np.ndarray) -> np.ndarray:
        return coords[::-1] / self.scale
    

    def draw_path(self, absolute_path: list[np.ndarray]):
        self.path = [self.absolute2display(point) for point in absolute_path]
    
    
    def update(self):
        padding = gray(50*(~self.owner.get_bitmap()).transpose().astype(np.uint8))
        walls = gray(255 - cv2.threshold(self.owner.array.transpose().astype(np.float32), 10, 50, cv2.THRESH_BINARY)[1].astype(np.uint8))
        self.image = pg.transform.scale_by(pg.surfarray.make_surface(padding-walls), self.scale)
        if self.path:
            if len(self.path) > 1:
                pg.draw.aalines(self.image, (0, 0, 255), False, self.path) # type: ignore