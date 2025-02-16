from .base_classes import DistanceMetric, Point
import numpy as np
from math import radians, sin, cos, sqrt, atan2

class HaversineDistance(DistanceMetric):
    """
    Implements the Haversine formula for calculating great-circle distances between points on a sphere.
    Specifically designed for calculating distances between latitude/longitude coordinates.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    
    def calculate(self, point1: Point, point2: Point) -> float:
        R = 6371  # Earth's radius in kilometers

        lat1, lon1 = radians(point1.latitude), radians(point1.longitude)
        lat2, lon2 = radians(point2.latitude), radians(point2.longitude)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

class EuclideanDistance(DistanceMetric):
    """
    Implements Euclidean distance between two points.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    
    def calculate(self, point1: Point, point2: Point) -> float:
        return sqrt(
            (point1.latitude - point2.latitude)**2 + 
            (point1.longitude - point2.longitude)**2
        )

def create_distance_matrix(points: list[Point], metric: DistanceMetric) -> np.ndarray:
    """
    Create a distance matrix for a list of points using the specified metric.
    
    Args:
        points: List of points
        metric: Distance metric to use
        
    Returns:
        Distance matrix as numpy array
        
    Time Complexity: O(n^2) where n is number of points
    Space Complexity: O(n^2)
    """
    n = len(points)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            distance = metric.calculate(points[i], points[j])
            matrix[i][j] = matrix[j][i] = distance
            
    return matrix