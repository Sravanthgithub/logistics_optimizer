from .base_classes import ClusteringAlgorithm, ClusterResult
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

class KMeansClustering(ClusteringAlgorithm):
    """
    K-Means clustering implementation.
    
    Time Complexity: O(k * n * i)
    where k is number of clusters, n is number of points, i is number of iterations
    
    Space Complexity: O(k + n)
    """
    
    def __init__(self, n_clusters: int = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray) -> ClusterResult:
        scaled_data = self.scaler.fit_transform(data)
        
        if self.n_clusters is None:
            self.n_clusters = self.detect_optimal_clusters(scaled_data)
            
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        labels = kmeans.fit_predict(scaled_data)
        centers = self.scaler.inverse_transform(kmeans.cluster_centers_)
        score = silhouette_score(scaled_data, labels)
        
        return ClusterResult(labels, centers, score)
    
    def detect_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
            
        return silhouette_scores.index(max(silhouette_scores)) + 2

class DBSCANClustering(ClusteringAlgorithm):
    """
    DBSCAN clustering implementation.
    
    Time Complexity: O(n * log n) with ball tree
    Space Complexity: O(n^2) in worst case
    """
    
    def __init__(self, eps: float = None, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray) -> ClusterResult:
        scaled_data = self.scaler.fit_transform(data)
        
        if self.eps is None:
            self.eps = self._estimate_eps(scaled_data)
            
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = dbscan.fit_predict(scaled_data)
        
        centers = np.array([
            scaled_data[labels == i].mean(axis=0) 
            for i in range(max(labels) + 1)
        ])
        centers = self.scaler.inverse_transform(centers)
        
        score = silhouette_score(scaled_data, labels) if len(np.unique(labels)) > 1 else 0
        
        return ClusterResult(labels, centers, score)
    
    def _estimate_eps(self, data: np.ndarray) -> float:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn_fit = nn.fit(data)
        distances, _ = nn_fit.kneighbors(data)
        distances = np.sort(distances[:, 1])
        return np.percentile(distances, 90)
    
    def detect_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        best_score = -1
        best_eps = None
        
        eps_range = np.linspace(0.1, 2.0, 20)
        
        for eps in eps_range:
            dbscan = DBSCAN(eps=eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(data)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
        
        self.eps = best_eps
        return len(np.unique(dbscan.labels_))

class HierarchicalClustering(ClusteringAlgorithm):
    """
    Hierarchical Agglomerative Clustering implementation.
    
    Time Complexity: O(n^3) in general case
    Space Complexity: O(n^2)
    """
    
    def __init__(self, n_clusters: int = None, linkage: str = 'ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.scaler = StandardScaler()
        
    def fit(self, data: np.ndarray) -> ClusterResult:
        scaled_data = self.scaler.fit_transform(data)
        
        if self.n_clusters is None:
            self.n_clusters = self.detect_optimal_clusters(scaled_data)
            
        hc = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        labels = hc.fit_predict(scaled_data)
        
        centers = np.array([
            scaled_data[labels == i].mean(axis=0) 
            for i in range(self.n_clusters)
        ])
        centers = self.scaler.inverse_transform(centers)
        
        score = silhouette_score(scaled_data, labels)
        
        return ClusterResult(labels, centers, score)
    
    def detect_optimal_clusters(self, data: np.ndarray, max_clusters: int = 10) -> int:
        best_score = -1
        optimal_k = 2
        
        for k in range(2, max_clusters + 1):
            hc = AgglomerativeClustering(n_clusters=k, linkage=self.linkage)
            labels = hc.fit_predict(data)
            score = calinski_harabasz_score(data, labels)
            
            if score > best_score:
                best_score = score
                optimal_k = k
                
        return optimal_k