from sklearn.cluster import Birch


class Model:
    def __init__(self, args):
        self.args = args
        self.threshold = getattr(args, "birch_threshold", 0.5)
        self.branching_factor = getattr(args, "birch_branching_factor", 50)
        self.n_clusters = getattr(args, "birch_n_clusters", None)
        self.compute_labels = getattr(args, "birch_compute_labels", True)
        self.birch = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=self.n_clusters,
            compute_labels=self.compute_labels,
        )

    def fit(self, X):
        return self.birch.fit(X)

    def predict(self, X):
        return self.birch.predict(X)

    def transform(self, X):
        return self.birch.transform(X)

    def to(self, device):
        return self
