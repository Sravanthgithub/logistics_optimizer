import os
from src.clustering import KMeansClustering, DBSCANClustering, HierarchicalClustering
from src.route_optimizers import (TwoOptOptimizer, NearestNeighborOptimizer, 
                                SimulatedAnnealingOptimizer, MinimalRouteOptimizer,
                                GraphTSPOptimizer)
from src.distance_metrics import HaversineDistance, EuclideanDistance, GeopyDistance, OSRMDistance
from src.visualization import MapVisualizer
from src.utils import load_config, load_points, timer_decorator, calculate_metrics, print_metrics
import time
from tabulate import tabulate
import webbrowser
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np

def compare_clustering_algorithms(data, visualizer, points, output_dir):
    """Compare different clustering algorithms and their performance metrics."""
    
    clustering_configs = {
        'KMeans': {
            'algorithm': KMeansClustering,
            'params': {'n_clusters': None}  # Will auto-detect
        },
        'DBSCAN': {
            'algorithm': DBSCANClustering,
            'params': {'eps': None, 'min_samples': 5}  # Will auto-detect eps
        },
        'Hierarchical': {
            'algorithm': HierarchicalClustering,
            'params': {'n_clusters': None, 'linkage': 'ward'}  # Will auto-detect
        }
    }
    
    results = []
    detailed_results = {}
    
    print("\nComparing Clustering Algorithms:")
    print("-" * 60)
    
    for name, config in clustering_configs.items():
        try:
            # Initialize and run clustering
            start_time = time.time()
            algo = config['algorithm'](**config['params'])
            cluster_result = algo.fit(data)
            end_time = time.time()
            
            # Calculate various clustering metrics
            n_clusters = len(set(cluster_result.labels[cluster_result.labels >= 0]))
            
            # Skip metrics if only one cluster
            if n_clusters <= 1:
                print(f"Warning: {name} produced only {n_clusters} clusters - skipping metrics")
                continue
                
            metrics = {
                'Silhouette Score': silhouette_score(data, cluster_result.labels),
                'Calinski-Harabasz Score': calinski_harabasz_score(data, cluster_result.labels),
                'Davies-Bouldin Score': davies_bouldin_score(data, cluster_result.labels),
                'Execution Time': round(end_time - start_time, 2),
                'Number of Clusters': n_clusters
            }
            
            # Save visualization for each algorithm
            vis_path = os.path.join(output_dir, f'clusters_{name.lower()}.html')
            visualizer.visualize_clusters(points, cluster_result, vis_path)
            
            results.append({
                'Algorithm': name,
                **metrics,
                'cluster_result': cluster_result
            })
            
            detailed_results[name] = {
                'metrics': metrics,
                'result': cluster_result,
                'vis_path': vis_path
            }
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    # Print comparison table
    table_data = [[
        r['Algorithm'],
        r['Number of Clusters'],
        round(r['Silhouette Score'], 3),
        round(r['Calinski-Harabasz Score'], 1),
        round(r['Davies-Bouldin Score'], 3),
        r['Execution Time']
    ] for r in results]
    
    print("\nClustering Algorithm Comparison:")
    print(tabulate(table_data,
                  headers=['Algorithm', 'Clusters', 'Silhouette', 'Calinski-Harabasz', 
                          'Davies-Bouldin', 'Time (sec)'],
                  tablefmt='grid'))
    
    # Analysis of results
    print("\nAnalysis of Clustering Results:")
    print("-" * 60)
    for name, result in detailed_results.items():
        metrics = result['metrics']
        print(f"\n{name}:")
        print(f"• Number of clusters: {metrics['Number of Clusters']}")
        print(f"• Silhouette Score: {metrics['Silhouette Score']:.3f} " +
              "(closer to 1 is better)")
        print(f"• Calinski-Harabasz Score: {metrics['Calinski-Harabasz Score']:.1f} " +
              "(higher is better)")
        print(f"• Davies-Bouldin Score: {metrics['Davies-Bouldin Score']:.3f} " +
              "(lower is better)")
        
        # Analyze cluster sizes
        labels = result['result'].labels
        sizes = [sum(labels == i) for i in range(max(labels) + 1)]
        print(f"• Cluster sizes: {sizes}")
        print(f"• Size variation: {np.std(sizes):.2f} (standard deviation)")
        
    # Select best algorithm based on combined metrics
    best_algo = max(results, key=lambda x: (
        x['Silhouette Score'] * 0.4 +  # Weight silhouette score more
        x['Calinski-Harabasz Score'] / 1000 * 0.3 +  # Normalize CH score
        (1 / x['Davies-Bouldin Score']) * 0.3  # Lower DB score is better
    ))
    
    print(f"\nRecommended algorithm: {best_algo['Algorithm']}")
    print(f"• Best balance of cluster quality metrics")
    print(f"• Generated {best_algo['Number of Clusters']} well-separated clusters")
    
    return best_algo['cluster_result']

def compare_route_optimizers(points, distance_metric):
    """Compare different route optimization algorithms and return the best one."""
    optimizers = {
        'minimal': MinimalRouteOptimizer(distance_metric),
        'two_opt': TwoOptOptimizer(distance_metric),
        'nearest_neighbor': NearestNeighborOptimizer(distance_metric),
        'simulated_annealing': SimulatedAnnealingOptimizer(
            distance_metric,
            initial_temp=100,
            cooling_rate=0.95,
            iterations=2000
        ),
        'graph_tsp': GraphTSPOptimizer(distance_metric)
    }
    
    results = []
    print(f"\nComparing route optimization algorithms for {len(points)} points:")
    print("-" * 60)
    
    for name, optimizer in optimizers.items():
        try:
            start_time = time.time()
            route = optimizer.optimize(points)
            end_time = time.time()
            
            result = {
                'Algorithm': name,
                'Distance (km)': round(route.distance, 2),
                'Time (sec)': round(end_time - start_time, 2),
                'optimizer': optimizer
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            continue
    
    results.sort(key=lambda x: x['Distance (km)'])
    
    table_data = [[r['Algorithm'], r['Distance (km)'], r['Time (sec)']] 
                 for r in results]
    print(tabulate(table_data, 
                  headers=['Algorithm', 'Distance (km)', 'Time (sec)'],
                  tablefmt='grid'))
    
    best_result = results[0]
    print(f"\nBest algorithm: {best_result['Algorithm']}")
    print(f"Distance: {best_result['Distance (km)']} km")
    print(f"Time: {best_result['Time (sec)']} seconds")
    
    return best_result['optimizer']

@timer_decorator
def main():
    # Load configuration
    config = load_config()
    
    # Create output directory if it doesn't exist
    os.makedirs(config['paths']['output'], exist_ok=True)
    
    # Load data
    points = load_points(config['paths']['data'])
    print(f"Loaded {len(points)} points")
    
    # Initialize distance metric
    distance_metric = {
        'haversine': HaversineDistance,
        'euclidean': EuclideanDistance,
        'geopy': GeopyDistance,
        'osrm': OSRMDistance
    }[config['distance_metric']['type']]()
    
    # Initialize visualizer
    visualizer = MapVisualizer()
    
    # Compare clustering algorithms
    data = [[p.latitude, p.longitude] for p in points]
    cluster_result = compare_clustering_algorithms(data, visualizer, points, 
                                                config['paths']['output'])
    
    # For each cluster, compare algorithms and use the best one
    print("\nOptimizing routes...")
    routes = {}
    cluster_results = {}
    
    for label in set(cluster_result.labels):
        if label >= 0:  # Ignore noise points (label = -1)
            cluster_points = [p for p, l in zip(points, cluster_result.labels) if l == label]
            print(f"\nOptimizing route for Cluster {label} ({len(cluster_points)} points)")
            
            best_optimizer = compare_route_optimizers(cluster_points, distance_metric)
            route = best_optimizer.optimize(cluster_points)
            routes[label] = route
            
            cluster_results[label] = {
                'points': len(cluster_points),
                'distance': route.distance,
                'algorithm': best_optimizer.__class__.__name__
            }
    
    # Print summary
    print("\nCluster Summary:")
    print("-" * 60)
    summary_data = [[
        f"Cluster {label}",
        result['points'],
        round(result['distance'], 2),
        result['algorithm']
    ] for label, result in cluster_results.items()]
    
    print(tabulate(summary_data,
                  headers=['Cluster', 'Points', 'Distance (km)', 'Best Algorithm'],
                  tablefmt='grid'))
    
    # Visualize final routes
    if config['visualization']['save_routes']:
        routes_path = os.path.join(config['paths']['output'], 'routes_final.html')
        visualizer.visualize_routes(routes, routes_path)
        print(f"\nSaved route visualization to {routes_path}")
    
    # Calculate and print metrics
    metrics = calculate_metrics(cluster_result, routes)
    print_metrics(metrics)

    # Open visualizations
    if config['visualization']['open_browser']:
        for name in ['kmeans', 'dbscan', 'hierarchical']:
            try:
                webbrowser.open(f"output/clusters_{name}.html")
            except:
                pass
        webbrowser.open('output/routes_final.html')

if __name__ == "__main__":
    main()