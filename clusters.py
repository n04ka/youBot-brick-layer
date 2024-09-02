import numpy as np
from numpy import pi as PI


class Detection:
    
    def __init__(self, coords: np.ndarray, rot: float, distance: float) -> None:
        self.coords = coords
        if -PI/2 < rot <= PI/2:
            self.rot = rot
        elif -PI/2 < rot:
            self.rot = rot + PI
        else:
            self.rot = rot - PI
        self.distance = distance


class Cluster:
    
    SIZE = 10
    
    def __init__(self) -> None:
        self.points: list[Detection] = []
        self.center: np.ndarray
        self.rot: float
        
        
    def drop_item(self) -> Detection:
        return self.points.pop(np.argmax([p.distance for p in self.points]))
        
        
    def add_item(self, item: Detection) -> None:
        np.seterr(all="ignore")
        self.points.append(item)
        while len(self.points) > self.SIZE: self.drop_item()
        self.center = np.median(np.array([p.coords for p in self.points]), axis=0)
        try:
            self.rot = np.nanmedian([float(p.rot) for p in self.points]) # type: ignore
        except RuntimeWarning:
            self.rot = np.median([float(p.rot) for p in self.points]) # type: ignore
    
        
class ClusterSystem:
    
    TOLERANCE = 0.2
    
    def __init__(self) -> None:
        self.clusters: list[Cluster] = []
        self.centers: np.ndarray = np.array([])
        self.restricted_areas_centers = []
        self.restricted_areas_radiuses = []
        
        
    def create_new_cluster(self) -> Cluster:
        new = Cluster()
        self.clusters.append(new)
        return new
    
    
    def restrict(self, center: np.ndarray, radius: float) -> None:
        self.restricted_areas_centers.append(center)
        self.restricted_areas_radiuses.append(radius)
    
    
    def add_item(self, item: Detection) -> None:
        
        if self.restricted_areas_centers:
            dist = np.linalg.norm(item.coords - np.array(self.restricted_areas_centers))
            if np.any(dist < np.array(self.restricted_areas_radiuses)):
                return
        
        if len(self.clusters) == 0:
            new = self.create_new_cluster()
            new.add_item(item)
            self.centers = np.array([new.center])
            return
            
        distances = np.linalg.norm(self.centers - item.coords, axis=1)
        min_i = np.argmin(distances)
        if distances[min_i] < self.TOLERANCE:
            self.clusters[min_i].add_item(item)
            self.centers[min_i] = self.clusters[min_i].center
            return
        
        new = self.create_new_cluster()
        new.add_item(item)
        self.centers = np.vstack((self.centers, new.center))
        
        
    def drop_clusters(self, center: np.ndarray, radius: float) -> None:
        exclusions = np.linalg.norm(self.centers - center, axis=1) < radius
        self.clusters = [cluster for cluster, excl in zip(self.clusters, exclusions) if not excl]
        self.centers = self.centers[~exclusions]
        
    
    def get_nearest(self, pos: np.ndarray) -> Cluster:
        if len(self.clusters) == 0:
            raise RuntimeError('No clusters in cluster system')
        min_i = np.argmin(np.linalg.norm(self.centers[:, :2] - pos, axis=1))
        cluster = self.clusters[min_i]
        return cluster
    
    
    def get_viz_data(self) -> tuple[np.ndarray, np.ndarray]:
        coords = np.array([point.coords for cluster in self.clusters for point in cluster.points])
        c_colors = np.array([np.random.random(3) for cluster in self.clusters])
        colors = np.array([color for cluster, color in zip(self.clusters, c_colors) for point in cluster.points])
        return coords, colors
    