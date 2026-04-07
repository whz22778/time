from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pickle
import warnings

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection_KMeans(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection_KMeans, self).__init__(args)
        self._win_len = None

    def plot_prediction_comparison(self, gt_arr, pred_arr, save_path, title, xlabel):
        gt_arr = np.asarray(gt_arr).astype(int)
        pred_arr = np.asarray(pred_arr).astype(int)
        x = np.arange(len(gt_arr))

        tp_idx = np.where((gt_arr == 1) & (pred_arr == 1))[0]
        fp_idx = np.where((gt_arr == 0) & (pred_arr == 1))[0]
        fn_idx = np.where((gt_arr == 1) & (pred_arr == 0))[0]

        plt.figure(figsize=(12, 3.6))
        plt.fill_between(
            x, 0, 1, where=(gt_arr == 1), step='mid',
            color='red', alpha=0.10
        )
        plt.step(x, gt_arr, where='mid', color='red', linewidth=1.8)
        plt.step(x, pred_arr, where='mid', color='green', linewidth=1.5, alpha=0.75)

        if tp_idx.size > 0:
            plt.scatter(
                tp_idx, np.full(tp_idx.shape, 1.05), s=24,
                facecolors='none', edgecolors='green', linewidths=1.2
            )
        if fp_idx.size > 0:
            plt.scatter(
                fp_idx, np.full(fp_idx.shape, 0.5), s=22,
                c='orange', marker='x', linewidths=1.2
            )
        if fn_idx.size > 0:
            plt.scatter(
                fn_idx, np.full(fn_idx.shape, 0.95), s=28,
                c='red', marker='v'
            )

        legend_handles = [
            Line2D([0], [0], color='red', linewidth=1.8, label='Ground Truth'),
            Line2D([0], [0], color='green', linewidth=1.5, label='Predicted'),
            Line2D(
                [0], [0], marker='o', color='green', markerfacecolor='none',
                markersize=5, linewidth=0, label='TP'
            ),
            Line2D(
                [0], [0], marker='x', color='orange',
                markersize=6, linewidth=0, label='FP'
            ),
            Line2D(
                [0], [0], marker='v', color='red',
                markersize=6, linewidth=0, label='FN'
            )
        ]

        plt.ylim([-0.1, 1.15])
        plt.xlabel(xlabel)
        plt.ylabel('Anomaly (0/1)')
        plt.title(title)
        plt.legend(handles=legend_handles, loc='upper right', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def _save_plt(self, test_energy, gt, pred, threshold, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        energy_norm = (test_energy - np.min(test_energy)) / (
            np.max(test_energy) - np.min(test_energy) + 1e-8
        )
        anomaly_idx = np.where(gt == 1)[0]
        pred_idx = np.where(pred == 1)[0]

        total_len = len(gt)
        zoom_len_max = 10000
        zoom_len = min(max(total_len // 5, 200), zoom_len_max, total_len)
        if anomaly_idx.size > 0:
            center_idx = int(np.median(anomaly_idx))
            zoom_start = max(0, center_idx - zoom_len // 2)
        else:
            zoom_start = 0
        zoom_end = min(total_len, zoom_start + zoom_len)
        zoom_start = max(0, zoom_end - zoom_len)

        anomaly_mask = (anomaly_idx >= zoom_start) & (anomaly_idx < zoom_end)
        pred_mask = (pred_idx >= zoom_start) & (pred_idx < zoom_end)
        anomaly_idx_zoom = anomaly_idx[anomaly_mask] - zoom_start
        pred_idx_zoom = pred_idx[pred_mask] - zoom_start

        try:
            self.plot_prediction_comparison(
                gt,
                pred,
                os.path.join(folder_path, 'prediction_vs_groundtruth.png'),
                'Prediction vs Ground Truth',
                'Time Index'
            )

            gt_zoom = gt[zoom_start:zoom_end]
            pred_zoom = pred[zoom_start:zoom_end]
            self.plot_prediction_comparison(
                gt_zoom,
                pred_zoom,
                os.path.join(folder_path, 'prediction_vs_groundtruth_zoom.png'),
                'Prediction vs Ground Truth (Zoom)',
                'Local Time Index'
            )
        except Exception:
            pass

        try:
            plt.figure(figsize=(12, 4))
            en_min, en_max = np.min(test_energy), np.max(test_energy)
            energy_norm_local = (test_energy - en_min) / (en_max - en_min + 1e-8)
            thr_norm = (threshold - en_min) / (en_max - en_min + 1e-8)

            plt.plot(energy_norm_local, label='Normalized Anomaly Score')
            plt.hlines(
                thr_norm, 0, len(energy_norm_local) - 1,
                colors='orange', linestyles='--', label='Threshold'
            )
            if anomaly_idx.size > 0:
                plt.scatter(
                    anomaly_idx, energy_norm_local[anomaly_idx], c='r', s=10,
                    label='Ground Truth Anomaly'
                )
            if pred_idx.size > 0:
                plt.scatter(
                    pred_idx, energy_norm_local[pred_idx], facecolors='none',
                    edgecolors='g', s=20, label='Predicted Anomaly'
                )
            plt.xlabel('Time Index')
            plt.ylabel('Normalized Score')
            plt.title('Anomaly Score with Threshold')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'anomaly_score_with_threshold.png'))
            plt.close()

            plt.figure(figsize=(12, 4))
            zoom_energy_local = energy_norm_local[zoom_start:zoom_end]
            plt.plot(zoom_energy_local, label='Normalized Anomaly Score (Zoom)')
            plt.hlines(
                thr_norm, 0, len(zoom_energy_local) - 1,
                colors='orange', linestyles='--', label='Threshold'
            )
            if anomaly_idx_zoom.size > 0:
                plt.scatter(
                    anomaly_idx_zoom, zoom_energy_local[anomaly_idx_zoom], c='r', s=14,
                    label='Ground Truth Anomaly'
                )
            if pred_idx_zoom.size > 0:
                plt.scatter(
                    pred_idx_zoom, zoom_energy_local[pred_idx_zoom], facecolors='none',
                    edgecolors='g', s=26, label='Predicted Anomaly'
                )
            plt.xlabel('Local Time Index')
            plt.ylabel('Normalized Score')
            plt.title('Anomaly Score with Threshold (Zoom)')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'anomaly_score_with_threshold_zoom.png'))
            plt.close()
        except Exception:
            pass

    def _save_pca_scatter(self, flat_windows, window_gt, window_pred, folder_path, filename):
        if flat_windows is None or window_gt is None or window_pred is None:
            return
        if len(flat_windows) == 0:
            return

        max_points = 5000
        n = len(flat_windows)
        if n > max_points:
            rng = np.random.default_rng(2021)
            idx = rng.choice(n, size=max_points, replace=False)
            flat_windows = flat_windows[idx]
            window_gt = window_gt[idx]
            window_pred = window_pred[idx]

        pca = PCA(n_components=2, random_state=0)
        coords = pca.fit_transform(flat_windows)

        tp = (window_gt == 1) & (window_pred == 1)
        fp = (window_gt == 0) & (window_pred == 1)
        fn = (window_gt == 1) & (window_pred == 0)
        tn = (window_gt == 0) & (window_pred == 0)

        plt.figure(figsize=(8, 6))
        if tn.any():
            plt.scatter(coords[tn, 0], coords[tn, 1], s=12, c='gray', alpha=0.5, label='TN')
        if tp.any():
            plt.scatter(coords[tp, 0], coords[tp, 1], s=18, c='green', alpha=0.8, label='TP')
        if fp.any():
            plt.scatter(coords[fp, 0], coords[fp, 1], s=18, c='orange', alpha=0.8, label='FP')
        if fn.any():
            plt.scatter(coords[fn, 0], coords[fn, 1], s=18, c='red', alpha=0.8, label='FN')

        plt.title('PCA Scatter (Anomaly Detection)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(folder_path, filename))
        plt.close()

    def _build_model(self):
        model_key = self.args.model
        model_map = self.model_dict.model_map if hasattr(self.model_dict, "model_map") else {}
        if model_key not in model_map:
            lower_map = {k.lower(): k for k in model_map.keys()}
            match = lower_map.get(str(model_key).lower())
            if match is None:
                # Fallback: rescan models directory in a case-insensitive way
                if os.path.exists('models'):
                    files = [f for f in os.listdir('models') if f.endswith('.py') and f != '__init__.py']
                    file_map = {os.path.splitext(f)[0].lower(): os.path.splitext(f)[0] for f in files}
                    match = file_map.get(str(model_key).lower())
            if match is None:
                available = ", ".join(sorted(model_map.keys()))
                raise NotImplementedError(
                    f"Model [{self.args.model}] not found in 'models' directory. Available: {available}"
                )
            model_key = match
        self.model_type = str(model_key).lower()
        model = self.model_dict[model_key](self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _flatten_windows(self, batch_x):
        if hasattr(batch_x, "detach"):
            batch_x = batch_x.detach().cpu().numpy()
        batch_x = np.asarray(batch_x)
        if batch_x.ndim == 2:
            batch_x = batch_x[:, :, None]
        return batch_x.reshape(batch_x.shape[0], -1)

    def _collect_flat_and_labels(self, loader, collect_labels, max_windows=None):
        flat_list = []
        point_labels = []
        window_labels = []
        win_len = None
        seen = 0

        for batch_x, batch_y in loader:
            if max_windows is not None and seen >= max_windows:
                break
            if max_windows is not None:
                remaining = max_windows - seen
                if hasattr(batch_x, "detach"):
                    batch_x_np = batch_x.detach().cpu().numpy()
                else:
                    batch_x_np = np.asarray(batch_x)
                if batch_x_np.ndim == 2:
                    batch_x_np = batch_x_np[:, :, None]
                if batch_x_np.shape[0] > remaining:
                    batch_x_np = batch_x_np[:remaining]
                    batch_x = batch_x[:remaining]
                    if hasattr(batch_y, "__len__"):
                        batch_y = batch_y[:remaining]
            else:
                batch_x_np = None

            if batch_x_np is not None:
                flat_list.append(batch_x_np.reshape(batch_x_np.shape[0], -1))
            else:
                flat_list.append(self._flatten_windows(batch_x))
            seen += flat_list[-1].shape[0]
            if collect_labels:
                if hasattr(batch_y, "detach"):
                    batch_y = batch_y.detach().cpu().numpy()
                batch_y = np.asarray(batch_y)
                if batch_y.ndim == 3:
                    batch_y = batch_y.squeeze(-1)
                if batch_y.ndim == 1:
                    window_lab = (batch_y > 0).astype(int)
                else:
                    window_lab = (batch_y.max(axis=1) > 0).astype(int)
                point_labels.append(batch_y.reshape(-1))
                window_labels.append(window_lab)
                if win_len is None:
                    win_len = int(batch_y.reshape(batch_y.shape[0], -1).shape[1])

        flat = np.concatenate(flat_list, axis=0) if flat_list else np.zeros((0, 0))
        if collect_labels:
            point_labels = np.concatenate(point_labels, axis=0)
            window_labels = np.concatenate(window_labels, axis=0)
            return flat, point_labels, window_labels, win_len
        return flat, None, None, None

    def _compute_window_scores(self, flat_windows):
        if self.model_type == 'kmeans':
            distances = self.model.transform(flat_windows)
            return np.min(distances, axis=1)
        if self.model_type == 'birch':
            distances = self.model.transform(flat_windows)
            return np.min(distances, axis=1)
        raise NotImplementedError(f"Model [{self.model_type}] does not support scoring.")

    def _dbscan_scores_and_pred(self, flat_windows):
        labels = self.model.fit_predict(flat_windows)
        window_pred = (labels == -1).astype(int)
        window_scores = window_pred.astype(float)
        return window_scores, window_pred

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        max_windows = None
        if self.model_type == 'dbscan' and str(getattr(self.args, "data", "")).upper() in ("NB15", "CIC"):
            max_windows = 50000
        train_x, _, _, _ = self._collect_flat_and_labels(
            train_loader, collect_labels=False, max_windows=max_windows
        )

        if self.model_type == 'kmeans':
            self.model.fit(train_x)
        elif self.model_type == 'birch':
            self.model.fit(train_x)
        elif self.model_type == 'dbscan':
            self.model.fit(train_x)
        else:
            raise NotImplementedError(f"Model [{self.model_type}] not supported.")

        with open(os.path.join(path, f'{self.model_type}.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

        return self.model

    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')
        train_loader = None
        if self.model_type != 'dbscan':
            _, train_loader = self._get_data(flag='train')

        if test:
            model_path = os.path.join('./checkpoints/' + setting, f'{self.model_type}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                print('Model checkpoint not found, run training first.')
                return

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        train_flat = None
        if train_loader is not None:
            train_flat, _, _, _ = self._collect_flat_and_labels(train_loader, collect_labels=False)
        test_flat, test_point_labels, test_window_labels, win_len = self._collect_flat_and_labels(
            test_loader, collect_labels=True
        )

        if win_len is None:
            win_len = self.args.seq_len
        self._win_len = win_len

        if self.model_type == 'dbscan':
            test_window_scores, window_pred = self._dbscan_scores_and_pred(test_flat)
            test_energy = np.repeat(test_window_scores, win_len)
            pred = np.repeat(window_pred, win_len)
            threshold = 0.5
            print("Threshold :", threshold)
        else:
            train_window_scores = self._compute_window_scores(train_flat)
            test_window_scores = self._compute_window_scores(test_flat)

            train_energy = np.repeat(train_window_scores, win_len)
            test_energy = np.repeat(test_window_scores, win_len)

            combined_energy = np.concatenate([train_energy, test_energy], axis=0)
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
            print("Threshold :", threshold)

            pred = (test_energy > threshold).astype(int)
        gt = test_point_labels.astype(int)


        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Before adjustment - Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        before_folder = os.path.join(folder_path, "before_adjustment")
        after_folder = os.path.join(folder_path, "after_adjustment")

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Before adjustment" + "\n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        self._save_plt(test_energy, gt, pred, threshold, before_folder)

        try:
            fpr, tpr, _ = roc_curve(gt, test_energy)
            roc_auc = auc(fpr, tpr)
            prec_curve, rec_curve, _ = precision_recall_curve(gt, test_energy)
            ap = average_precision_score(gt, test_energy)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label='ROC curve (AUC = %0.4f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'roc_curve.png'))
            plt.close()

            plt.figure()
            plt.plot(rec_curve, prec_curve, color='blue', lw=2,
                     label='PR curve (AP = %0.4f)' % ap)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'pr_curve.png'))
            plt.close()
            f.write('AUC : {:0.4f}, Average Precision (AP) : {:0.4f}\n'.format(roc_auc, ap))
        except Exception as e:
            f.write('Curve generation failed: ' + str(e) + '\n')

        pred_before = pred.copy()
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("After adjustment - Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f.write("After adjustment" + "\n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        self._save_plt(test_energy, gt, pred, threshold, after_folder)

        if test_window_labels is not None:
            window_pred_before = pred_before.reshape(-1, win_len).max(axis=1)
            window_pred_after = pred.reshape(-1, win_len).max(axis=1)
            self._save_pca_scatter(
                test_flat,
                test_window_labels,
                window_pred_before,
                folder_path,
                'pca_anomaly_scatter_before.png'
            )
            self._save_pca_scatter(
                test_flat,
                test_window_labels,
                window_pred_after,
                folder_path,
                'pca_anomaly_scatter_after.png'
            )

        f.write('\n')
        f.close()
        return
