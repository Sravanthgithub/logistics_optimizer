from .base_classes import RouteOptimizer, Point, Route, DistanceMetric
from .distance_metrics import create_distance_matrix
import numpy as np
from typing import List, Tuple
import time
from itertools import permutations
import networkx as nx

class TwoOptOptimizer(RouteOptimizer):
    """
    2-opt local search algorithm for route optimization.
    Repeatedly finds pairs of edges that can be swapped to reduce total distance.
    
    Time Complexity: O(n^2 * i) where n is number of points and i is iterations
    Space Complexity: O(n^2) for distance matrix
    """
    
    def __init__(self, metric: DistanceMetric):
        self.metric = metric
    
    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        if start_point:
            points = [start_point] + points
            
        distances = create_distance_matrix(points, self.metric)
        n = len(points)
        current_route = list(range(n))
        best_distance = self._calculate_route_distance(current_route, distances)
        
        improvement = True
        while improvement:
            improvement = False
            
            for i in range(1, n-2):
                for j in range(i+1, n):
                    new_route = self._two_opt_swap(current_route, i, j)
                    new_distance = self._calculate_route_distance(new_route, distances)
                    
                    if new_distance < best_distance:
                        current_route = new_route
                        best_distance = new_distance
                        improvement = True
                        break
                if improvement:
                    break
        
        return Route(
            points=[points[i] for i in current_route],
            distance=best_distance,
            sequence=current_route
        )
    
    def _two_opt_swap(self, route: List[int], i: int, k: int) -> List[int]:
        """Perform 2-opt swap by reversing the segment between positions i and k."""
        return route[:i] + route[i:k+1][::-1] + route[k+1:]
    
    def _calculate_route_distance(self, route: List[int], distances: np.ndarray) -> float:
        """Calculate total distance of a route using the distance matrix."""
        total = sum(distances[route[i]][route[i-1]] for i in range(1, len(route)))
        return total + distances[route[0]][route[-1]]  # Add return to start

class NearestNeighborOptimizer(RouteOptimizer):
    """
    Nearest Neighbor algorithm for route optimization.
    Greedily selects the closest unvisited point at each step.
    
    Time Complexity: O(n^2) where n is number of points
    Space Complexity: O(n^2) for distance matrix
    """
    
    def __init__(self, metric: DistanceMetric):
        self.metric = metric
    
    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        distances = create_distance_matrix(points, self.metric)
        n = len(points)
        
        # Start from first point if no start point specified
        current = 0 if not start_point else points.index(start_point)
        unvisited = set(range(n))
        unvisited.remove(current)
        
        route = [current]
        total_distance = 0
        
        while unvisited:
            next_point = min(unvisited, key=lambda x: distances[current][x])
            total_distance += distances[current][next_point]
            route.append(next_point)
            unvisited.remove(next_point)
            current = next_point
            
        # Add return to start
        total_distance += distances[route[-1]][route[0]]
        
        return Route(
            points=[points[i] for i in route],
            distance=total_distance,
            sequence=route
        )

class BruteForceOptimizer(RouteOptimizer):
    """
    Brute force algorithm trying all possible permutations.
    Only practical for very small sets of points (n <= 10).
    
    Time Complexity: O(n!) where n is number of points
    Space Complexity: O(n^2) for distance matrix
    """
    
    def __init__(self, metric: DistanceMetric):
        self.metric = metric
    
    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        if len(points) > 10:
            raise ValueError("Brute force optimization not recommended for more than 10 points")
            
        distances = create_distance_matrix(points, self.metric)
        n = len(points)
        
        best_route = None
        best_distance = float('inf')
        
        for perm in permutations(range(n)):
            distance = self._calculate_route_distance(perm, distances)
            if distance < best_distance:
                best_distance = distance
                best_route = perm
                
        return Route(
            points=[points[i] for i in best_route],
            distance=best_distance,
            sequence=list(best_route)
        )
    
    def _calculate_route_distance(self, route: tuple, distances: np.ndarray) -> float:
        """Calculate total distance of a route using the distance matrix."""
        total = sum(distances[route[i]][route[i-1]] for i in range(1, len(route)))
        return total + distances[route[0]][route[-1]]

class SimulatedAnnealingOptimizer(RouteOptimizer):
    """
    Simulated Annealing algorithm for route optimization.
    Uses probabilistic acceptance of worse solutions to escape local optima.
    
    Time Complexity: O(n^2 * i) where n is number of points and i is iterations
    Space Complexity: O(n^2) for distance matrix
    """
    
    def __init__(self, metric: DistanceMetric, 
                 initial_temp: float = 100,
                 cooling_rate: float = 0.95,
                 iterations: int = 1000):
        self.metric = metric
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
    
    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        distances = create_distance_matrix(points, self.metric)
        n = len(points)
        
        # Initialize with random route
        current_route = np.random.permutation(n).tolist()
        current_distance = self._calculate_route_distance(current_route, distances)
        
        best_route = current_route.copy()
        best_distance = current_distance
        
        temperature = self.initial_temp
        
        for _ in range(self.iterations):
            # Generate neighbor by swapping two random positions
            i, j = np.random.randint(0, n, 2)
            neighbor_route = current_route.copy()
            neighbor_route[i], neighbor_route[j] = neighbor_route[j], neighbor_route[i]
            
            neighbor_distance = self._calculate_route_distance(neighbor_route, distances)
            
            # Calculate acceptance probability
            delta = neighbor_distance - current_distance
            if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                current_route = neighbor_route
                current_distance = neighbor_distance
                
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            temperature *= self.cooling_rate
        
        return Route(
            points=[points[i] for i in best_route],
            distance=best_distance,
            sequence=best_route
        )
    
    def _calculate_route_distance(self, route: List[int], distances: np.ndarray) -> float:
        """Calculate total distance of a route using the distance matrix."""
        total = sum(distances[route[i]][route[i-1]] for i in range(1, len(route)))
        return total + distances[route[0]][route[-1]]

class GraphTSPOptimizer(RouteOptimizer):
    """
    Graph-based TSP solver using multiple approaches:
    1. Christofides algorithm for metric TSP (approximation ratio 1.5)
    2. Minimum Spanning Tree + DFS approach
    3. Dynamic Programming for small graphs
    """
    
    def __init__(self, metric):
        self.metric = metric

    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        """
        Choose the best method based on graph size:
        - For n ≤ 15: Use Dynamic Programming (optimal)
        - For n > 15: Use Christofides if metric, else MST+DFS
        """
        if len(points) <= 15:
            return self._solve_dp(points, start_point)
        else:
            return self._solve_christofides(points, start_point)

    def _create_graph(self, points: List[Point]) -> nx.Graph:
        """Create complete graph with distances as weights."""
        G = nx.Graph()
        
        # Add nodes
        for i, point in enumerate(points):
            G.add_node(i, pos=(point.latitude, point.longitude))
            
        # Add edges with weights
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                weight = self.metric.calculate(points[i], points[j])
                G.add_edge(i, j, weight=weight)
                
        return G

    def _solve_dp(self, points: List[Point], start_point: Point = None) -> Route:
        """
        Solve TSP using Dynamic Programming.
        Time Complexity: O(n^2 * 2^n)
        Space Complexity: O(n * 2^n)
        """
        n = len(points)
        all_points = (1 << n) - 1
        
        # dp[mask][pos] = min cost to visit all cities in mask ending at pos
        dp = defaultdict(lambda: defaultdict(lambda: float('inf')))
        parent = defaultdict(dict)
        
        # Base case: start from node 0
        dp[1][0] = 0
        
        # For each subset of cities
        for mask in range(1, 1 << n):
            # For each possible last city
            for pos in range(n):
                if not (mask & (1 << pos)):
                    continue
                    
                prev_mask = mask ^ (1 << pos)
                # Try to come to pos from all other cities in the subset
                for prev in range(n):
                    if not (prev_mask & (1 << prev)):
                        continue
                        
                    cost = dp[prev_mask][prev] + self.metric.calculate(points[prev], points[pos])
                    if cost < dp[mask][pos]:
                        dp[mask][pos] = cost
                        parent[mask][pos] = prev
        
        # Find best end city
        final_cost = float('inf')
        final_city = 0
        for end in range(n):
            cost = dp[all_points][end] + self.metric.calculate(points[end], points[0])
            if cost < final_cost:
                final_cost = cost
                final_city = end
                
        # Reconstruct path
        path = []
        curr_mask = all_points
        curr_city = final_city
        while curr_mask:
            path.append(curr_city)
            new_city = parent[curr_mask][curr_city]
            curr_mask ^= (1 << curr_city)
            curr_city = new_city
            
        path.append(0)  # Return to start
        
        return Route(
            points=[points[i] for i in path],
            distance=final_cost,
            sequence=path
        )

    def _solve_christofides(self, points: List[Point], start_point: Point = None) -> Route:
        """
        Solve TSP using Christofides algorithm.
        Guaranteed 1.5-approximation for metric TSP.
        Time Complexity: O(n^3)
        Space Complexity: O(n^2)
        """
        G = self._create_graph(points)
        
        # 1. Find minimum spanning tree
        mst = nx.minimum_spanning_tree(G, weight='weight')
        
        # 2. Find odd-degree vertices
        odd_vertices = [v for v, d in mst.degree() if d % 2 == 1]
        
        # 3. Find minimum-weight perfect matching on odd vertices
        subgraph = G.subgraph(odd_vertices)
        matching = nx.min_weight_matching(subgraph, weight='weight', maxcardinality=True)
        
        # 4. Combine MST and matching
        eulerian_multigraph = nx.MultiGraph(mst)
        for v1, v2 in matching:
            eulerian_multigraph.add_edge(v1, v2, weight=G[v1][v2]['weight'])
            
        # 5. Find Eulerian circuit
        euler_circuit = list(nx.eulerian_circuit(eulerian_multigraph))
        
        # 6. Convert to Hamiltonian cycle (shortcut)
        visited = set()
        path = []
        total_distance = 0
        
        for v1, v2 in euler_circuit:
            if v1 not in visited:
                path.append(v1)
                visited.add(v1)
                
        path.append(path[0])  # Return to start
        
        # Calculate total distance
        for i in range(len(path) - 1):
            total_distance += G[path[i]][path[i+1]]['weight']
            
        return Route(
            points=[points[i] for i in path],
            distance=total_distance,
            sequence=path
        )

class MinimalRouteOptimizer(RouteOptimizer):
    """
    A focused implementation to find shortest route through all points.
    Uses nearest neighbor with 2-opt improvement as it provides good 
    balance between speed and solution quality.
    """
    
    def __init__(self, metric):
        self.metric = metric

    def optimize(self, points: List[Point], start_point: Point = None) -> Route:
        """
        Find shortest route visiting all points and returning to start.
        Implements the abstract method from RouteOptimizer base class.
        
        Args:
            points: List of points to visit
            start_point: Optional starting point
            
        Returns:
            Route object containing optimized sequence and total distance
        """
        # Create distance matrix for quick lookups
        distances = self._create_distance_matrix(points)
        
        # Find initial route using nearest neighbor
        route = self._get_initial_route(distances)
        
        # Improve route using 2-opt swaps
        improved_route = self._improve_route(route, distances)
        
        # Calculate total distance
        total_distance = self._calculate_total_distance(improved_route, distances)
        
        return Route(
            points=[points[i] for i in improved_route],
            distance=total_distance,
            sequence=improved_route
        )

    def _create_distance_matrix(self, points: List[Point]) -> np.ndarray:
        """Create matrix of distances between all points."""
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distance = self.metric.calculate(points[i], points[j])
                distances[i][j] = distances[j][i] = distance
                
        return distances

    def _get_initial_route(self, distances: np.ndarray) -> List[int]:
        """Get initial route using nearest neighbor approach."""
        n = len(distances)
        unvisited = set(range(1, n))
        route = [0]  # Start with first point
        current = 0
        
        while unvisited:
            # Find nearest unvisited point
            next_point = min(unvisited, key=lambda x: distances[current][x])
            route.append(next_point)
            unvisited.remove(next_point)
            current = next_point
            
        return route

    def _improve_route(self, route: List[int], distances: np.ndarray) -> List[int]:
        """Improve route using 2-opt swaps."""
        n = len(route)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    if self._check_swap_improvement(route, i, j, distances):
                        route[i:j+1] = reversed(route[i:j+1])
                        improved = True
                        break
                if improved:
                    break
                    
        return route

    def _check_swap_improvement(self, route: List[int], i: int, j: int, 
                              distances: np.ndarray) -> bool:
        """Check if 2-opt swap would improve route distance."""
        n = len(route)
        # Calculate current segment distances
        current_distance = (distances[route[i-1]][route[i]] + 
                          distances[route[j]][route[j+1]])
        # Calculate new segment distances after swap
        new_distance = (distances[route[i-1]][route[j]] + 
                       distances[route[i]][route[j+1]])
        
        return new_distance < current_distance

    def _calculate_total_distance(self, route: List[int], 
                                distances: np.ndarray) -> float:
        """Calculate total distance of route."""
        total = sum(distances[route[i]][route[i-1]] 
                   for i in range(1, len(route)))
        # Add return to start
        total += distances[route[-1]][route[0]]
        return total