import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
from threading import Thread


pg.init()
SCALE = 0.1
SCREEN = pg.display.set_mode((0, 0), pg.FULLSCREEN)
CLOCK = pg.time.Clock()


class Map:
    
    def __init__(self, pos: tuple, shape: tuple, scale: float) -> None:
        self.pos = np.array(pos, dtype=np.int16)
        self.array = np.zeros(shape, dtype=np.uint16)
        self.scale = scale
    
    
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
    
    
    def update(self):
        self.image = pg.transform.scale_by(pg.surfarray.make_surface(gray(self.owner.array.transpose())), self.scale)


world_map = Map((0, 0), (0, 0), SCALE)
robot = Robot()
display = MapDisplay(world_map)

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
        for event in pg.event.get():
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == pg.K_q):
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                x, y = event.pos
                print(display.display2absolute(np.array([x-display.rect.topleft[0], y]))) # type: ignore
                
        displays.update()
        robots.update()
        
        SCREEN.fill((128, 128, 128))
        robots.draw(display.image) # type: ignore
        displays.draw(SCREEN)
        font = pg.font.SysFont(None, 24)
        x, y = display.display2absolute(np.array(robot.rect.center)) # type: ignore
        surf = font.render(f'Robot pos: {x:.2f}, {y:.2f}', True, (0, 0, 0))
        SCREEN.blit(surf, (20, 20))
        CLOCK.tick(60)
        pg.display.flip()
    pg.quit()


displays: pg.sprite.GroupSingle = pg.sprite.GroupSingle(display)
robots: pg.sprite.GroupSingle = pg.sprite.GroupSingle(robot)
update_thread = Thread(target=update_map, daemon=True)
update_thread.start()
visualize()
update_thread.join()