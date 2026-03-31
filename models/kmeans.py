from sklearn.cluster import KMeans


class Model:
    def __init__(self, args):
        self.args = args
        self.n_clusters = getattr(args, "kmeans_n_clusters", 10)
        self.max_iter = getattr(args, "kmeans_max_iter", 300)
        self.tol = getattr(args, "kmeans_tol", 1e-4)
        self.init = getattr(args, "kmeans_init", "k-means++")
        self.random_state = getattr(args, "seed", 2)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init="auto",
        )

    def fit(self, X):
        return self.kmeans.fit(X)

    def predict(self, X):
        return self.kmeans.predict(X)

    def transform(self, X):
        return self.kmeans.transform(X)

    def to(self, device):
        return self
