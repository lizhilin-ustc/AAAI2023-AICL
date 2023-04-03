import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE
from config.config_thumos import Config, parse_args, class_dict
from models.model import AICL

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

np.set_printoptions(threshold=np.inf)

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):                #　　0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        self.net = AICL(config)
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)

        # parameters
        self.best_mAP = -1 # init
        self.step = 0
        self.total_loss_per_epoch = 0


    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).cuda()

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments


    def calculate_all_losses1(self, contrast_pairs, contrast_pairs_r,contrast_pairs_f, cas_top, label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2):
        self.contrastive_criterion = ContrastiveLoss()
        loss_contrastive = self.contrastive_criterion(contrast_pairs) + self.contrastive_criterion(contrast_pairs_r) + self.contrastive_criterion(contrast_pairs_f)

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = 0.5 * F.mse_loss(action_flow, action_rgb) + 0.5 * F.mse_loss(action_rgb, action_flow)
        action_consistent_loss = 0.5 * F.mse_loss(actionness1, actionness2) + 0.5 * F.mse_loss(actionness2, actionness1)

        cost = base_loss + class_agnostic_loss  + 5*modality_consistent_loss + 0.01*loss_contrastive + 0.1*action_consistent_loss

        return cost

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))

            print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  best_map={:5.2f}".format(
                    epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0


    def forward_pass(self, _data):
        cas, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.net(_data)

        combined_cas = misc_utils.instance_selection_function(torch.softmax(cas.detach(), -1),
                                                              action_flow.permute(0, 2, 1).detach(),
                                                              action_rgb.permute(0, 2, 1))


        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        # _, topk_indices1 = torch.topk(combined_cas, r, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2


    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):

            for _data, _label, temp_anno, _, _ in self.train_loader:

                batch_size = _data.shape[0]
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                # forward pass
                cas_top, topk_indices, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = self.forward_pass(_data)

                # calcualte pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)

                # losses
                cost = self.calculate_all_losses1(contrast_pairs, contrast_pairs_r,contrast_pairs_f, cas_top, _label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2)

                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)



def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
