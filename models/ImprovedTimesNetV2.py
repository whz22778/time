import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class FeatureInteractionLayer(nn.Module):
    """
    Build pairwise cross features for the top-k "important" input channels.

    Compared with the previous ImprovedTimesNet implementation, we do NOT assume
    the "first k channels" are important. Instead we pick top-k by a simple
    data-driven score each forward pass (no gradients through the selection).
    """

    def __init__(self, k=6, mode: str = "dynamic"):
        super().__init__()
        self.k = int(k)
        self.mode = str(mode).lower()

    def forward(self, x):
        # x: [B, L, D]
        b, l, d = x.shape
        k = min(self.k, d)
        if k <= 1:
            return x

        if self.mode == "fixed":
            # Use the first k channels (best when features are pre-sorted by MI importance).
            topk_feat = x[:, :, :k]
        else:
            # Importance score per feature channel (stable + cheap).
            # score: [D]
            score = x.detach().abs().mean(dim=(0, 1))
            topk_idx = torch.topk(score, k=k, largest=True, sorted=False).indices  # [k]

            # Gather: [B, L, k]
            idx = topk_idx.view(1, 1, k).expand(b, l, k)
            topk_feat = torch.gather(x, dim=2, index=idx)

        # Pairwise products -> cross_dim = k*(k-1)/2
        crosses = []
        for i in range(k):
            for j in range(i + 1, k):
                crosses.append(topk_feat[:, :, i] * topk_feat[:, :, j])
        cross = torch.stack(crosses, dim=-1)  # [B, L, cross_dim]
        return torch.cat([x, cross], dim=-1)


class FeatureReweighting(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1, 1, dim))

    def forward(self, x):
        return x * self.w


class ClusterGuidedFeature(nn.Module):
    """
    Learnable cluster centers over the original input feature space.
    Distances to centers are appended as additional features.
    """

    def __init__(self, n_clusters=5, input_dim=64):
        super().__init__()
        self.n_clusters = int(n_clusters)
        self.input_dim = int(input_dim)
        self.cluster_centers = nn.Parameter(torch.randn(self.n_clusters, self.input_dim))

    def forward(self, x):
        # x: [B, L, D] where the first input_dim channels are the original features
        # Follow the "KMeans.transform" style: distances for every time step.
        b, l, _ = x.shape
        x_flat = x[:, :, : self.input_dim].reshape(b * l, self.input_dim)  # [B*L, input_dim]
        distances = torch.cdist(x_flat, self.cluster_centers)  # [B*L, n_clusters]
        dist_seq = distances.reshape(b, l, self.n_clusters)  # [B, L, n_clusters]
        return torch.cat([x, dist_seq], dim=-1)


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        b, t, n = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x

            out = out.reshape(b, length // period, period, n).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(b, -1, n)
            res.append(out[:, : (self.seq_len + self.pred_len), :])

        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, t, n, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class Model(nn.Module):
    """
    Improved TimesNet (V2):
    - Fixes the channel mismatch in the previous ImprovedTimesNet implementation by
      embedding the *augmented* feature dimension (original + interactions + cluster distances).
    - Uses a data-driven top-k feature selection for interaction features.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # -------- Feature augmentation (in input space) --------
        enc_in = int(configs.enc_in)
        k = min(int(getattr(configs, "feature_k", 6)), enc_in)
        cross_dim = k * (k - 1) // 2
        n_clusters = int(getattr(configs, "n_clusters", 5))
        self.aug_dim = enc_in
        if self.configs.asn >> 2 & 1:
            self.aug_dim += cross_dim
        if self.configs.asn >> 1 & 1:
            self.aug_dim += n_clusters

        interaction_mode = getattr(configs, "interaction_topk", "dynamic")
        self.feature_interaction = FeatureInteractionLayer(k=k, mode=interaction_mode)
        self.cluster_feature = ClusterGuidedFeature(n_clusters=n_clusters, input_dim=enc_in)
        self.feature_reweighting = FeatureReweighting(dim=self.aug_dim)

        # Embed augmented features (this is the key fix vs. ImprovedTimesNet.py)
        self.enc_embedding = DataEmbedding(self.aug_dim, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Optional: KMeans init cluster centers directly from disk (no exp/ changes needed).
        # This runs once at model construction time.
        self._maybe_init_cluster_centers_from_disk(enc_in=enc_in)

        # -------- Heads (same as TimesNet) --------
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name in ["imputation", "anomaly_detection"]:
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def _maybe_init_cluster_centers_from_disk(self, enc_in: int):
        """
        If `--cluster_init kmeans`, fit KMeans on `root_path/train.csv` (normal data),
        then write centers into `self.cluster_feature.cluster_centers`.

        This lets us enable the feature without touching `exp/exp_anomaly_detection.py`.
        """
        if self.task_name != "anomaly_detection":
            return

        if getattr(self.configs, "cluster_init", "random") != "kmeans":
            return

        root_path = getattr(self.configs, "root_path", None)
        if not root_path:
            return

        try:
            import os
            import numpy as np
            import pandas as pd
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except Exception:
            return

        train_path = os.path.join(root_path, "train.csv")
        if not os.path.exists(train_path):
            return

        try:
            train_df = pd.read_csv(train_path)
        except Exception:
            return

        x = train_df.values[:, 1:]
        x = np.nan_to_num(x)
        if x.ndim != 2 or x.shape[1] != enc_in:
            return

        scaler = StandardScaler()
        scaler.fit(x)
        x_scaled = scaler.transform(x)

        n = x_scaled.shape[0]
        max_n = int(getattr(self.configs, "cluster_kmeans_samples", 20000))
        max_n = max(1, min(max_n, n))
        seed = int(getattr(self.configs, "cluster_kmeans_seed", 2021))
        rs = np.random.RandomState(seed)
        x_fit = x_scaled[rs.choice(n, size=max_n, replace=False)] if max_n < n else x_scaled

        n_clusters = int(getattr(self.configs, "n_clusters", self.cluster_feature.n_clusters))
        max_iter = int(getattr(self.configs, "cluster_kmeans_max_iter", 300))
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=seed,
                n_init=10,
                max_iter=max_iter,
            )
            kmeans.fit(x_fit)
        except Exception:
            return

        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        with torch.no_grad():
            if centers.shape == self.cluster_feature.cluster_centers.shape:
                self.cluster_feature.cluster_centers.copy_(centers)

        if getattr(self.configs, "cluster_freeze", False):
            self.cluster_feature.cluster_centers.requires_grad_(False)

    def _augment(self, x_enc):
        if self.configs.asn >> 2 & 1:
            x_enc = self.feature_interaction(x_enc)
        if self.configs.asn >> 1 & 1:
            x_enc = self.cluster_feature(x_enc)
        if self.configs.asn & 1:
            x_enc = self.feature_reweighting(x_enc)
        return x_enc

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        x_enc = self._augment(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        x_enc = self._augment(x_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        x_enc = self._augment(x_enc)

        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out.add(means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_enc = self._augment(x_enc)

        enc_out = self.enc_embedding(x_enc, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None
