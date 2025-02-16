from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class Point:
    latitude: float
    longitude: float

@dataclass
class Route:
    points: List[Point]
    distance: float
    sequence: List[int]

@dataclass
class ClusterResult:
    labels: np.ndarray
    centers: np.ndarray
    score: float

class ClusteringAlgorithm(ABC):
    """
    Abstract base class for clustering algorithms.
    Defines the interface that all clustering implementations must follow.
    """
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> ClusterResult:
        """
        Fit the clustering algorithm to the data.
        
        Args:
            data: Array of shape (n_samples, n_features)
            
        Returns:
            ClusterResult containing labels, centers and clustering score
        """
        pass
    
    @abstractmethod
    def detect_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        """
        Detect the optimal number of clusters for the given data.
        
        Args:
            data: Array of shape (n_samples, n_features)
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            Optimal number of clusters
        """
        pass

class RouteOptimizer(ABC):
    """
    Abstract base class for route optimization algorithms.
    All route optimization implementations must implement this interface.
    """
    
    @abstractmethod
    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        """
        Optimize the route through the given points.
        
        Args:
            points: List of points to visit
            start_point: Optional starting point
            
        Returns:
            Route object containing optimized sequence and total distance
            
        Time Complexity: Must be specified in implementing classes
        Space Complexity: Must be specified in implementing classes
        """
        pass

class DistanceMetric(ABC):
    """
    Abstract base class for distance metrics.
    All distance metric implementations must implement this interface.
    """
    
    @abstractmethod
    def calculate(self, point1: Point, point2: Point) -> float:
        """
        Calculate distance between two points.
        
        Args:
            point1: First point
            point2: Second point
            
        Returns:
            Distance between points
        """
        pass