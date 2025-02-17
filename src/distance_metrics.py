from .base_classes import DistanceMetric, Point
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from geopy.distance import geodesic
import requests
import time
from functools import lru_cache

class HaversineDistance(DistanceMetric):
    """
    Implements the Haversine formula for calculating great-circle distances.
    
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

class GeopyDistance(DistanceMetric):
    """
    Uses geopy's geodesic distance calculation.
    More accurate than Haversine for geographical calculations.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    
    def calculate(self, point1: Point, point2: Point) -> float:
        return geodesic(
            (point1.latitude, point1.longitude),
            (point2.latitude, point2.longitude)
        ).kilometers

class OSRMDistance(DistanceMetric):
    """
    Uses OSRM for real road network distances.
    Includes caching and rate limiting.
    """
    
    def __init__(self):
        self.base_url = "http://router.project-osrm.org/route/v1/driving"
        self.last_request_time = 0
        self.min_request_interval = 1.0  # seconds
        self.cache = {}

    def calculate(self, point1: Point, point2: Point) -> float:
        """Calculate actual driving distance using OSRM."""
        # Check cache first
        cache_key = (point1, point2)
        reverse_cache_key = (point2, point1)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        if reverse_cache_key in self.cache:
            return self.cache[reverse_cache_key]
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        try:
            url = f"{self.base_url}/{point1.longitude},{point1.latitude};"
            url += f"{point2.longitude},{point2.latitude}"
            
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('code') != 'Ok':
                # Fallback to Haversine if OSRM fails
                distance = HaversineDistance().calculate(point1, point2)
            else:
                # Convert distance from meters to kilometers
                distance = data['routes'][0]['distance'] / 1000
            
            # Cache the result
            self.cache[cache_key] = distance
            return distance
            
        except Exception as e:
            print(f"OSRM API error: {str(e)}, falling back to Haversine")
            distance = HaversineDistance().calculate(point1, point2)
            self.cache[cache_key] = distance
            return distance
        
        finally:
            self.last_request_time = time.time()

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