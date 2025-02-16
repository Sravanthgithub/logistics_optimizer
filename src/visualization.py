import folium
from folium import plugins
import branca.colormap as cm
import numpy as np
from typing import List, Dict
from .base_classes import Point, Route, ClusterResult

class MapVisualizer:
    """
    Handles visualization of clusters and routes on an interactive map.
    """
    
    def __init__(self, center_lat: float = None, center_lon: float = None):
        self.center_lat = center_lat
        self.center_lon = center_lon
        
    def create_base_map(self, points: List[Point] = None) -> folium.Map:
        """
        Create a base map centered on the mean of points or specified center.
        """
        if points and not (self.center_lat and self.center_lon):
            self.center_lat = np.mean([p.latitude for p in points])
            self.center_lon = np.mean([p.longitude for p in points])
            
        return folium.Map(
            location=[self.center_lat, self.center_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
    
    def visualize_clusters(self, 
                          points: List[Point], 
                          cluster_result: ClusterResult,
                          save_path: str = 'clusters.html') -> None:
        """
        Visualize clustering results on an interactive map.
        """
        m = self.create_base_map(points)
        
        # Create color map
        n_clusters = len(np.unique(cluster_result.labels))
        colors = self._generate_colors(n_clusters)
        
        # Plot points
        for point, label in zip(points, cluster_result.labels):
            if label >= 0:  # Ignore noise points (label = -1)
                folium.CircleMarker(
                    location=[point.latitude, point.longitude],
                    radius=8,
                    color=colors[label],
                    fill=True,
                    popup=f'Cluster {label}'
                ).add_to(m)
        
        # Plot cluster centers
        for i, center in enumerate(cluster_result.centers):
            folium.CircleMarker(
                location=[center[0], center[1]],
                radius=10,
                color='red',
                fill=True,
                popup=f'Center {i}'
            ).add_to(m)
        
        m.save(save_path)
    
    def visualize_routes(self,
                        routes: Dict[int, Route],
                        save_path: str = 'routes.html') -> None:
        """
        Visualize optimized routes on an interactive map.
        """
        m = self.create_base_map()
        
        colors = self._generate_colors(len(routes))
        
        for cluster_id, route in routes.items():
            # Create route coordinates
            route_coords = [[p.latitude, p.longitude] for p in route.points]
            
            # Add route line
            folium.PolyLine(
                route_coords,
                color=colors[cluster_id],
                weight=2,
                opacity=0.8
            ).add_to(m)
            
            # Add markers for each point
            for i, point in enumerate(route.points):
                folium.CircleMarker(
                    location=[point.latitude, point.longitude],
                    radius=6,
                    color=colors[cluster_id],
                    fill=True,
                    popup=f'Stop {i} (Cluster {cluster_id})'
                ).add_to(m)
        
        m.save(save_path)
    
    @staticmethod
    def _generate_colors(n: int) -> List[str]:
        """Generate n distinct colors."""
        return [f'#{hash(str(i))%0xFFFFFF:06x}' for i in range(n)]