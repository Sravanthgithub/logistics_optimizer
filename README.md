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

## Clustering Algorithms

### K-Means Clustering
**Description**: Partitions points into k clusters by minimizing within-cluster variances.

**When to Use**:
- When clusters are expected to be roughly spherical
- When clusters are expected to be of similar size
- When the number of clusters is known or can be estimated
- For large-scale clustering problems

**Pros**:
- Simple and fast
- Works well with large datasets
- Clusters are easily interpretable

**Cons**:
- Needs number of clusters specified
- Sensitive to outliers
- Assumes spherical clusters
- May converge to local optima

### DBSCAN (Density-Based Spatial Clustering)
**Description**: Groups points based on density, identifying clusters of arbitrary shape.

**When to Use**:
- When clusters have irregular shapes
- When the number of clusters is unknown
- When data contains noise/outliers
- When clusters have varying densities

**Pros**:
- Doesn't require predefined number of clusters
- Can find arbitrarily shaped clusters
- Robust to outliers
- Can identify noise points

**Cons**:
- Sensitive to parameters (eps and min_samples)
- Struggles with varying density clusters
- May have trouble with high-dimensional data
- Can be slower than K-means

### Hierarchical Clustering
**Description**: Builds a tree of clusters, either by merging (agglomerative) or splitting (divisive).

**When to Use**:
- When hierarchical relationship between clusters is important
- When visualizing the clustering process is helpful
- For smaller datasets
- When cluster size varies significantly

**Pros**:
- Provides hierarchical representation
- No need to specify number of clusters beforehand
- More flexible than K-means
- Good for hierarchical data

**Cons**:
- Computationally intensive (O(n³))
- Not suitable for large datasets
- Can be sensitive to noise
- Results can be harder to interpret

## Route Optimization Algorithms

### Minimal Route Optimizer
**Description**: Combines Nearest Neighbor approach with 2-opt improvement.

**When to Use**:
- For medium-sized clusters (20-100 points)
- When balanced performance is needed
- When computation time is a concern
- For general-purpose route optimization

**Complexity**:
- Time: O(n²)
- Space: O(n²)

### Two-Opt Optimizer
**Description**: Local search algorithm that repeatedly swaps edges to improve route.

**When to Use**:
- For small to medium clusters
- When solution quality is prioritized over speed
- When routes have obvious crossovers
- For fine-tuning existing routes

**Complexity**:
- Time: O(n² * i) where i is iterations
- Space: O(n²)

### Graph TSP Optimizer
**Description**: Graph-based approach using NetworkX, implements Christofides algorithm.

**When to Use**:
- For metric TSP problems
- When theoretical guarantees are important
- When clusters are small enough (≤15 points) for exact solutions
- When balanced performance is needed

**Complexity**:
- Time: O(n³)
- Space: O(n²)

### Simulated Annealing Optimizer
**Description**: Probabilistic optimization that can escape local optima.

**When to Use**:
- For complex landscapes with many local optima
- When longer computation time is acceptable
- When other methods get stuck in local optima
- For large-scale optimization problems

**Complexity**:
- Time: O(n² * i) where i is iterations
- Space: O(n²)

### Nearest Neighbor Optimizer
**Description**: Greedy algorithm that always chooses closest unvisited point.

**When to Use**:
- When fast solutions are needed
- For initial route approximation
- As part of hybrid algorithms
- When solution quality is less critical

**Complexity**:
- Time: O(n²)
- Space: O(n²)

## Evaluation Metrics

### Clustering Metrics
- **Silhouette Score**: Measures how similar points are to their own cluster vs other clusters (range: -1 to 1, higher is better)
- **Calinski-Harabasz Score**: Ratio of between-cluster to within-cluster dispersion (higher is better)
- **Davies-Bouldin Score**: Average similarity measure of each cluster with its most similar cluster (lower is better)

### Route Metrics
- **Total Distance**: Sum of distances between consecutive points in route
- **Computation Time**: Time taken to find solution
- **Route Feasibility**: Checks if route visits all points exactly once

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

## Usage

1. Place your data CSV file in the `data/` directory
2. Adjust configuration in `config/config.yaml`
3. Run:
```bash
python main.py
```

4. Check results in `output/` directory

## Algorithm Selection Guidelines

### For Clustering
1. Start with K-means if:
   - Clusters are expected to be roughly spherical
   - Quick results are needed
   - Data size is large

2. Use DBSCAN if:
   - Data has noise/outliers
   - Clusters have irregular shapes
   - Number of clusters is unknown

3. Use Hierarchical if:
   - Hierarchical structure is important
   - Dataset is small
   - Cluster visualization is needed

### For Route Optimization
1. Start with Minimal Route Optimizer if:
   - Balanced performance is needed
   - Clusters are medium-sized

2. Use Graph TSP if:
   - Clusters are small (≤15 points)
   - Optimal solution is needed
   - Theoretical guarantees are important

3. Try Simulated Annealing if:
   - Other methods get stuck in local optima
   - Longer computation time is acceptable

## Dependencies

See `requirements.txt` for complete list of dependencies.

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

![KMeans Routes](https://github.com/user-attachments/assets/07e9b897-0937-445f-b91d-6abacc11a693)

```
