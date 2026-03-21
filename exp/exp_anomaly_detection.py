from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import torch.multiprocessing
import matplotlib.pyplot as plt

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach()
                true = batch_x.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                outputs = self.model(batch_x, None, None, None)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        # 常检测metric
        # gt, pred = adjustment(gt, pred) 

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        try:
            # 计算基于连续异常分数的 ROC / PR 指标，适合观察阈值变化下的整体性能。
            fpr, tpr, _ = roc_curve(gt, test_energy)
            roc_auc = auc(fpr, tpr)
            prec_curve, rec_curve, _ = precision_recall_curve(gt, test_energy)
            ap = average_precision_score(gt, test_energy)

            # 保存 ROC 曲线。
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

            # 保存 PR 曲线。
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

            # 将异常分数归一化到 [0, 1]，方便和阈值、标签一起展示。
            energy_norm = (test_energy - np.min(test_energy)) / (
                np.max(test_energy) - np.min(test_energy) + 1e-8
            )
            anomaly_idx = np.where(gt == 1)[0]
            pred_idx = np.where(pred == 1)[0]

            # 额外截取一段局部区间用于放大显示：
            # 优先围绕真实异常区域，如果没有异常点则展示前 20% 的测试序列。
            # 同时限制一个最大窗口长度，避免测试集过长时局部图仍然过于拥挤。
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

            # 全量测试集异常分数图。
            plt.figure(figsize=(12, 4))
            plt.plot(energy_norm, label='Normalized Anomaly Score')
            if anomaly_idx.size > 0:
                plt.scatter(
                    anomaly_idx, energy_norm[anomaly_idx], c='r', s=10,
                    label='Ground Truth Anomaly'
                )
            if pred_idx.size > 0:
                plt.scatter(
                    pred_idx, energy_norm[pred_idx], facecolors='none',
                    edgecolors='g', s=20, label='Predicted Anomaly'
                )
            plt.xlabel('Time Index')
            plt.ylabel('Normalized Score')
            plt.title('Anomaly Scores on Test Set')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'anomaly_score_timeseries.png'))
            plt.close()

            # 局部放大异常分数图，便于观察异常附近的分数变化。
            plt.figure(figsize=(12, 4))
            zoom_energy = energy_norm[zoom_start:zoom_end]
            plt.plot(zoom_energy, label='Normalized Anomaly Score (Zoom)')
            if anomaly_idx_zoom.size > 0:
                plt.scatter(
                    anomaly_idx_zoom, zoom_energy[anomaly_idx_zoom], c='r', s=14,
                    label='Ground Truth Anomaly'
                )
            if pred_idx_zoom.size > 0:
                plt.scatter(
                    pred_idx_zoom, zoom_energy[pred_idx_zoom], facecolors='none',
                    edgecolors='g', s=26, label='Predicted Anomaly'
                )
            plt.xlabel('Local Time Index')
            plt.ylabel('Normalized Score')
            plt.title('Anomaly Scores on Test Set (Zoom)')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(folder_path, 'anomaly_score_timeseries_zoom.png'))
            plt.close()

            f.write('AUC : {:0.4f}, Average Precision (AP) : {:0.4f}\n'.format(roc_auc, ap))

            try:
                # 全量二值预测与真实标签对比图。
                plt.figure(figsize=(12, 3))
                plt.step(range(len(gt)), gt, where='mid', label='Ground Truth', color='red')
                plt.step(range(len(pred)), pred, where='mid', label='Predicted', color='green', alpha=0.7)
                plt.ylim([-0.1, 1.1])
                plt.xlabel('Time Index')
                plt.ylabel('Anomaly (0/1)')
                plt.title('Prediction vs Ground Truth')
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, 'prediction_vs_groundtruth.png'))
                plt.close()

                # 局部放大的预测与真实标签对比图。
                plt.figure(figsize=(12, 3))
                gt_zoom = gt[zoom_start:zoom_end]
                pred_zoom = pred[zoom_start:zoom_end]
                plt.step(range(len(gt_zoom)), gt_zoom, where='mid', label='Ground Truth', color='red')
                plt.step(range(len(pred_zoom)), pred_zoom, where='mid', label='Predicted', color='green', alpha=0.7)
                plt.ylim([-0.1, 1.1])
                plt.xlabel('Local Time Index')
                plt.ylabel('Anomaly (0/1)')
                plt.title('Prediction vs Ground Truth (Zoom)')
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(folder_path, 'prediction_vs_groundtruth_zoom.png'))
                plt.close()
            except Exception:
                pass

            try:
                # 全量异常分数 + 阈值图，用于直观看到阈值与预测位置。
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

                # 局部放大异常分数 + 阈值图，便于查看阈值附近的误报和漏报。
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
        except Exception as e:
            f.write('Curve generation failed: ' + str(e) + '\n')
        f.write('\n')
        f.close()
        return
