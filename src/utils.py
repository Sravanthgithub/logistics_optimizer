import pandas as pd
import yaml
from typing import List, Dict, Any
from .base_classes import Point
import time
from functools import wraps

def load_config(path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_points(file_path: str) -> List[Point]:
    """Load points from CSV file."""
    df = pd.read_csv(file_path)
    return [Point(row.Latitude, row.Longitude) for _, row in df.iterrows()]

def timer_decorator(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def calculate_metrics(cluster_result, routes):
    """Calculate and return various metrics about the clustering and routing."""
    metrics = {
        'num_clusters': len(set(cluster_result.labels)),
        'silhouette_score': cluster_result.score,
        'total_distance': sum(route.distance for route in routes.values()),
        'points_per_cluster': {},
        'route_distances': {}
    }
    
    for cluster_id, route in routes.items():
        metrics['points_per_cluster'][cluster_id] = len(route.points)
        metrics['route_distances'][cluster_id] = route.distance
        
    return metrics

def print_metrics(metrics: Dict):
    """Print metrics in a formatted way."""
    print("\n=== Optimization Results ===")
    print(f"Number of clusters: {metrics['num_clusters']}")
    print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"Total route distance: {metrics['total_distance']:.2f} km")
    print("\nPoints per cluster:")
    for cluster_id, count in metrics['points_per_cluster'].items():
        print(f"  Cluster {cluster_id}: {count} points")
    print("\nRoute distances:")
    for cluster_id, distance in metrics['route_distances'].items():
        print(f"  Cluster {cluster_id}: {distance:.2f} km")