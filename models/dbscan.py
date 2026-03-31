from sklearn.cluster import DBSCAN


class Model:
    def __init__(self, args):
        self.args = args
        self.eps = getattr(args, "dbscan_eps", 0.5)
        self.min_samples = getattr(args, "dbscan_min_samples", 5)
        self.metric = getattr(args, "dbscan_metric", "euclidean")
        self.algorithm = getattr(args, "dbscan_algorithm", "auto")
        self.leaf_size = getattr(args, "dbscan_leaf_size", 30)
        self.n_jobs = getattr(args, "dbscan_n_jobs", None)
        self.dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )

    def fit(self, X):
        return self.dbscan.fit(X)

    def fit_predict(self, X):
        return self.dbscan.fit_predict(X)

    def to(self, device):
        return self
