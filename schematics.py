import numpy as np
from matplotlib.pyplot import imshow
from cv2 import polylines
from json import loads


class Scheme:
    
    def __init__(self, json: dict) -> None:
        self.json = json
        self.interval = json['interval']
        self.total = json['total']
        self.points = np.array([[brick['x'], brick['y'], brick['z']] for brick in json['bricks'].values()])
        self.shapes = np.array([[brick['l'], brick['w'], brick['h']] for brick in json['bricks'].values()])
        self.rots = np.array([brick['r'] for brick in json['bricks'].values()])
        self.limits = np.min(self.points, axis=0), np.max(self.points, axis=0)
    
    
    def show(self):
        xlimits = self.limits[0][0]-self.interval*100, self.limits[1][0]+self.interval*100
        ylimits = self.limits[0][1]-self.interval*100, self.limits[1][1]+self.interval*100
        shape = [round((lim[1] - lim[0])//self.interval) for lim in [xlimits, ylimits]]

        self.absolute2img = lambda coords: tuple(np.round((np.array(coords) - np.array([xlimits[0], ylimits[0]]))//self.interval).astype(int))
        
        img = np.zeros((*shape[::-1], 3))
        
        for brick in self.json['bricks'].values():
            rect = np.array([   [-brick['l']/2, -brick['w']/2],
                                [-brick['l']/2, brick['w']/2],
                                [brick['l']/2,  brick['w']/2],
                                [brick['l']/2,  -brick['w']/2]])
            rot_matrix = np.array([ [np.cos(brick['r']), -np.sin(brick['r'])],
                                    [np.sin(brick['r']), np.cos(brick['r'])]])
            rotated = np.dot(rect, rot_matrix) + np.array([brick['x'], brick['y']]).reshape((-1, 1, 2))
            polylines(img, [np.array(self.absolute2img(coords)) for coords in rotated], True, (255, 0, 0), 1)
        imshow(img, extent=[*xlimits, *ylimits[::-1]])
    
    
    def iterator(self):
        for i in range(self.total):
            yield self.points[i], self.rots[i], self.shapes[i]
        
    
def open_scheme(file_name: str):
    with open(file_name, 'r') as file:
        scheme = Scheme(loads(file.read()))
    return scheme