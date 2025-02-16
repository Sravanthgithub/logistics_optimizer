# Logistics Route Optimizer

A Python-based solution for optimizing logistics routes using various clustering and route optimization algorithms.

## Project Structure
```
logistics_optimizer/
│
├── src/
│   ├── __init__.py               # Package initialization
│   ├── base_classes.py           # Abstract base classes and data structures
│   ├── clustering.py             # Clustering algorithm implementations
│   ├── distance_metrics.py       # Distance calculation implementations
│   ├── route_optimizers.py       # Route optimization algorithms
│   ├── visualization.py          # Map visualization utilities
│   └── utils.py                  # Helper functions
│
├── config/
│   └── config.yaml               # Configuration settings
│
├── data/
│   └── Mojro Data For Assignment.csv    # Input data
│
├── output/                       # Generated files (created automatically)
│   ├── clusters.html            # Cluster visualization
│   └── routes.html              # Route visualization
│
├── requirements.txt             # Project dependencies
├── main.py                      # Main execution script
└── README.md                    # This file
```

## Features

- Multiple clustering algorithms:
  - K-Means
  - DBSCAN
  - Hierarchical Clustering

- Multiple route optimization algorithms:
  - Minimal Route Optimizer (Nearest Neighbor with 2-opt improvement)
  - Two-Opt Optimizer
  - Graph-based TSP Optimizer
  - Simulated Annealing
  - Nearest Neighbor

- Distance metrics:
  - Haversine (for geographic coordinates)
  - Euclidean

- Interactive visualizations using Folium maps

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to customize:
- Clustering algorithm and parameters
- Route optimization algorithm
- Distance metric
- Input/output paths
- Visualization settings

Example configuration:
```yaml
clustering:
  algorithm: kmeans
  params:
    n_clusters: null  # null for auto-detection
    random_state: 42
    
route_optimization:
  algorithm: minimal  # or two_opt, graph_tsp, simulated_annealing, nearest_neighbor
  
distance_metric:
  type: haversine  # or euclidean
```

## Usage

1. Place your data CSV file in the `data/` directory

2. Run the optimizer:
```bash
python main.py
```

3. Check the results:
- Console output shows comparative performance metrics
- Visualizations are saved in the `output/` directory
  - `clusters.html`: Shows clustered points
  - `routes.html`: Shows optimized routes

## Algorithm Details

### Clustering Algorithms
- **K-Means**: Partitions points into k clusters
- **DBSCAN**: Density-based clustering, good for irregular shapes
- **Hierarchical**: Agglomerative clustering with various linkage options

### Route Optimization
- **Minimal Route Optimizer**: 
  - Combines Nearest Neighbor with 2-opt improvement
  - Good balance of speed and solution quality
  - Time Complexity: O(n²)

- **Two-Opt Optimizer**: 
  - Local search algorithm
  - Repeatedly swaps edges to improve route
  - Time Complexity: O(n² * i) where i is iterations

- **Graph TSP**: 
  - Graph-based approach using NetworkX
  - Implements Christofides algorithm for metric TSP
  - Guaranteed 1.5-approximation for metric spaces

- **Simulated Annealing**: 
  - Probabilistic optimization
  - Can escape local optima
  - Good for complex landscapes

## Notes on TSP

The Traveling Salesman Problem (TSP) is NP-hard, meaning:
- No known polynomial-time algorithm for optimal solution
- All practical algorithms are approximations
- Different algorithms may perform better for different data distributions

The program automatically compares all available algorithms and selects the best one for each cluster based on:
- Route distance
- Computation time
- Cluster size and characteristics

## Output Example

```
Comparing route optimization algorithms for 64 points:
+---------------------+-----------------+--------------+
| Algorithm           |   Distance (km) |   Time (sec) |
+=====================+=================+==============+
| minimal             |            0.22 |         0.08 |
| two_opt             |            0.22 |         2.11 |
| graph_tsp           |            0.22 |         0.05 |
| nearest_neighbor    |            0.27 |         0.00 |
| simulated_annealing |            0.47 |         0.10 |
+---------------------+-----------------+--------------+
```

## Contributing

To add new algorithms:
1. Implement the appropriate interface from `base_classes.py`
2. Add the implementation to the respective module
3. Update the configuration options in `config.yaml`

## Dependencies

- numpy
- pandas
- scikit-learn
- folium
- pyyaml
- branca
- networkx
- tabulate

## Example Output
```
Loaded 137 points

Comparing Clustering Algorithms:
------------------------------------------------------------

Clustering Algorithm Comparison:
+--------------+------------+--------------+---------------------+------------------+--------------+
| Algorithm    |   Clusters |   Silhouette |   Calinski-Harabasz |   Davies-Bouldin |   Time (sec) |
+==============+============+==============+=====================+==================+==============+
| KMeans       |          3 |        0.664 |               993.6 |            0.486 |         0.07 |
+--------------+------------+--------------+---------------------+------------------+--------------+
| DBSCAN       |          7 |       -0.21  |                12   |            1.437 |         0.01 |
+--------------+------------+--------------+---------------------+------------------+--------------+
| Hierarchical |         10 |        0.514 |              1757.5 |            0.674 |         0.02 |
+--------------+------------+--------------+---------------------+------------------+--------------+

Analysis of Clustering Results:
------------------------------------------------------------

KMeans:
• Number of clusters: 3
• Silhouette Score: 0.664 (closer to 1 is better)
• Calinski-Harabasz Score: 993.6 (higher is better)
• Davies-Bouldin Score: 0.486 (lower is better)
• Cluster sizes: [np.int64(28), np.int64(64), np.int64(45)]
• Size variation: 14.70 (standard deviation)

DBSCAN:
• Number of clusters: 7
• Silhouette Score: -0.210 (closer to 1 is better)
• Calinski-Harabasz Score: 12.0 (higher is better)
• Davies-Bouldin Score: 1.437 (lower is better)
• Cluster sizes: [np.int64(5), np.int64(6), np.int64(7), np.int64(11), np.int64(5), np.int64(11), np.int64(8)]
• Size variation: 2.38 (standard deviation)

Hierarchical:
• Number of clusters: 10
• Silhouette Score: 0.514 (closer to 1 is better)
• Calinski-Harabasz Score: 1757.5 (higher is better)
• Davies-Bouldin Score: 0.674 (lower is better)
• Cluster sizes: [np.int64(10), np.int64(22), np.int64(16), np.int64(16), np.int64(20), np.int64(18), np.int64(19), np.int64(4), np.int64(5), np.int64(7)]
• Size variation: 6.28 (standard deviation)

Recommended algorithm: KMeans
• Best balance of cluster quality metrics
• Generated 3 well-separated clusters

Optimizing routes...

Optimizing route for Cluster 0 (28 points)

Comparing route optimization algorithms for 28 points:
------------------------------------------------------------
+---------------------+-----------------+--------------+
| Algorithm           |   Distance (km) |   Time (sec) |
+=====================+=================+==============+
| two_opt             |            0.1  |         0.04 |
+---------------------+-----------------+--------------+
| graph_tsp           |            0.11 |         0.01 |
+---------------------+-----------------+--------------+
| minimal             |            0.12 |         0    |
+---------------------+-----------------+--------------+
| nearest_neighbor    |            0.12 |         0    |
+---------------------+-----------------+--------------+
| simulated_annealing |            0.15 |         0.05 |
+---------------------+-----------------+--------------+

Best algorithm: two_opt
Distance: 0.1 km
Time: 0.04 seconds

Optimizing route for Cluster 1 (64 points)

Comparing route optimization algorithms for 64 points:
------------------------------------------------------------
+---------------------+-----------------+--------------+
| Algorithm           |   Distance (km) |   Time (sec) |
+=====================+=================+==============+
| minimal             |            0.22 |         0.06 |
+---------------------+-----------------+--------------+
| two_opt             |            0.22 |         1.63 |
+---------------------+-----------------+--------------+
| graph_tsp           |            0.22 |         0.03 |
+---------------------+-----------------+--------------+
| nearest_neighbor    |            0.27 |         0    |
+---------------------+-----------------+--------------+
| simulated_annealing |            0.38 |         0.07 |
+---------------------+-----------------+--------------+

Best algorithm: minimal
Distance: 0.22 km
Time: 0.06 seconds

Optimizing route for Cluster 2 (45 points)

Comparing route optimization algorithms for 45 points:
------------------------------------------------------------
+---------------------+-----------------+--------------+
| Algorithm           |   Distance (km) |   Time (sec) |
+=====================+=================+==============+
| minimal             |            0.1  |         0.01 |
+---------------------+-----------------+--------------+
| two_opt             |            0.1  |         0.43 |
+---------------------+-----------------+--------------+
| graph_tsp           |            0.1  |         0.01 |
+---------------------+-----------------+--------------+
| nearest_neighbor    |            0.11 |         0    |
+---------------------+-----------------+--------------+
| simulated_annealing |            0.15 |         0.07 |
+---------------------+-----------------+--------------+

Best algorithm: minimal
Distance: 0.1 km
Time: 0.01 seconds

Cluster Summary:
------------------------------------------------------------
+-----------+----------+-----------------+-----------------------+
| Cluster   |   Points |   Distance (km) | Best Algorithm        |
+===========+==========+=================+=======================+
| Cluster 0 |       28 |            0.1  | TwoOptOptimizer       |
+-----------+----------+-----------------+-----------------------+
| Cluster 1 |       64 |            0.22 | MinimalRouteOptimizer |
+-----------+----------+-----------------+-----------------------+
| Cluster 2 |       45 |            0.1  | MinimalRouteOptimizer |
+-----------+----------+-----------------+-----------------------+

Saved route visualization to output/routes_final.html

=== Optimization Results ===
Number of clusters: 3
Silhouette score: 0.686
Total route distance: 0.42 km

Points per cluster:
  Cluster 0: 28 points
  Cluster 1: 64 points
  Cluster 2: 45 points

Route distances:
  Cluster 0: 0.10 km
  Cluster 1: 0.22 km
  Cluster 2: 0.10 km
main execution time: 3.82 seconds
```