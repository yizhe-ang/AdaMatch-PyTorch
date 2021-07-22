import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from dassl.data import DataManager
from dassl.data.transforms import build_transform
from dassl.engine import TRAINER_REGISTRY, TrainerXU
from dassl.engine.trainer import SimpleNet
from dassl.metrics import compute_accuracy
from dassl.optim.optimizer import build_optimizer
from dassl.utils.torchtools import count_num_param, load_pretrained_weights

from .utils import (
    ExponentialMovingAverage,
    ExponentialMovingAverageModule,
    MovingAverage,
    cosine_schedule,
    unnormalize_img,
)


class AdaMatchNet(SimpleNet):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__(cfg, model_cfg, num_classes, **kwargs)

        # Keep a running estimate of source / target label distributions
        self.label_dist_s = ExponentialMovingAverage(
            (num_classes,), init_value=1 / num_classes
        )
        self.label_dist_t = MovingAverage((num_classes,), init_value=1 / num_classes)


@TRAINER_REGISTRY.register()
class AdaMatch(TrainerXU):
    """AdaMatch: A Unified Approach to Semi-Supervised Learning
    and Domain Adaptation

    https://arxiv.org/abs/2106.04732
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.conf_thre = cfg.TRAINER.ADAMATCH.CONF_THRE

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.ADAMATCH.STRONG_TRANSFORMS) > 0

    def build_data_loader(self):
        cfg = self.cfg

        # Initialize training transformations
        # Initialize default / weak transforms
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        # Initialize strong transforms
        choices = cfg.TRAINER.ADAMATCH.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)

        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)

        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader  # is `None`
        self.test_loader = self.dm.test_loader

        self.num_classes = self.dm.num_classes

    def build_model(self):
        cfg = self.cfg

        print("Building model")
        self.model = AdaMatchNet(cfg, cfg.MODEL, self.num_classes)

        # Keep EMA of model parameters for evaluation
        self.ema_momentum = cfg.TRAINER.ADAMATCH.EMA_MOMENTUM
        if self.ema_momentum:
            self.model = ExponentialMovingAverageModule(self.model, self.ema_momentum)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print("# params: {:,}".format(count_num_param(self.model)))
        self.optim = build_optimizer(self.model, cfg.OPTIM)

        # Use custom lr_scheduler
        num_batches = len(self.train_loader_u)
        self.max_iter = self.max_epoch * num_batches

        lr_lambda = partial(
            cosine_schedule,
            max_iter=self.max_iter,
            decay=cfg.OPTIM.LR_SCHEDULER_DECAY,
        )
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.optim,
            lr_lambda,
        )

        self.register_model("model", self.model, self.optim, self.sched)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        # Accuracy of masked predictions
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)
        # Raw accuracy
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()
        keep_rate = mask.sum() / mask.numel()

        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}

        return output

    def forward_backward(self, batch_s, batch_t):
        n_iter = self.epoch * self.num_batches + self.batch_idx

        parsed_data = self.parse_batch_train(batch_s, batch_t)
        input_s_w, input_s_s, label_s, input_t_w, input_t_s, label_t = parsed_data

        # Log sample training images
        if n_iter == 0:
            self._write_train_imgs(
                {
                    "source_weak_imgs": input_s_w,
                    "source_strong_imgs": input_s_s,
                    "target_weak_imgs": input_t_w,
                    "target_strong_imgs": input_t_s,
                }
            )

        n_source_w = input_s_w.size(0)
        n_source = n_source_w * 2

        # [N_s*2 + N_t*2, C, H, W]
        input_all = torch.cat([input_s_w, input_s_s, input_t_w, input_t_s], dim=0)
        # [N_s*2, C, H, W]
        input_s = torch.cat([input_s_w, input_s_s], dim=0)

        # Generate two different outputs of source input
        logits_all = self.model(input_all)  # [N_all, K]
        logits_s_prime = logits_all[:n_source]  # [N_s*2, K]

        self._disable_batchnorm_tracking(self.model)
        logits_s_dprime = self.model(input_s)  # [N_s*2, K]
        self._enable_batchnorm_tracking(self.model)

        # 1. Perform Random Logit Interpolation
        lamb = torch.rand_like(logits_s_prime)
        logits_s = (lamb * logits_s_prime) + ((1 - lamb) * logits_s_dprime)
        logits_s_w, logits_s_s = logits_s.chunk(2, dim=0)  # [N_s, K]

        # 2. Distribution Alignment
        logits_t_w, logits_t_s = logits_all[n_source:].chunk(2, dim=0)  # [N_t, K]

        prob_s_w = F.softmax(logits_s_w, dim=1)  # [N_s, K]
        prob_t_w = F.softmax(logits_t_w, dim=1)  # [N_t, K]

        # Align the target label distribution to that of the source
        # Update running estimates of label distributions
        model = self.model.model if self.ema_momentum else self.model
        label_dist_s = model.label_dist_s(prob_s_w.detach().mean(dim=0))
        label_dist_t = model.label_dist_t(prob_t_w.detach().mean(dim=0))
        expected_ratio = (1e-6 + label_dist_s) / (1e-6 + label_dist_t)  # [K]
        pseudo_t_w_unnorm = prob_t_w * expected_ratio  # [N_t, K]

        # Normalize back to a valid probability dist
        pseudo_t_w = pseudo_t_w_unnorm / pseudo_t_w_unnorm.sum(dim=1, keepdims=True)
        # Generate soft pseudo-labels for target loss computation; stop gradient
        pseudo_t_w = pseudo_t_w.detach()

        # 3. Relative Confidence Threshold
        mean_max_conf = prob_s_w.detach().max(dim=1).values.mean()
        rel_conf_thre = self.conf_thre * mean_max_conf
        # Filter for target pseudo-labels with high confidence
        mask_t = (pseudo_t_w.max(dim=1).values >= rel_conf_thre).float()  # [N_t]

        # 4. Loss Function
        loss_source = 0.5 * (
            F.cross_entropy(logits_s_w, label_s) + F.cross_entropy(logits_s_s, label_s)
        )

        loss_target = F.cross_entropy(
            logits_t_s, pseudo_t_w.argmax(dim=1), reduction="none"
        )  # [N_t]
        loss_target = (loss_target * mask_t).mean()

        mu = self._compute_loss_target_weight(n_iter, self.max_iter)
        loss = loss_source + (mu * loss_target)

        self.model_backward_and_update(loss)
        self.model.update()  # Update EMA

        with torch.no_grad():
            # Evaluate pseudo-labels' accuracy
            y_t_pred_stats = self.assess_y_pred_quality(
                pseudo_t_w.argmax(dim=1), label_t, mask_t
            )

            # Logging
            loss_summary = {
                "loss_s": loss_source.item(),
                "acc_s": compute_accuracy(logits_s, label_s.repeat(2))[0].item(),
                "loss_t": loss_target.item(),
                "y_t_pred_acc_raw": y_t_pred_stats["acc_raw"],
                "y_t_pred_acc_thre": y_t_pred_stats["acc_thre"],
                "y_t_pred_keep": y_t_pred_stats["keep_rate"],
            }
            self.write_scalar("train/loss_t_weight", mu, n_iter)
            self.write_scalar("train/rel_conf_thre", rel_conf_thre, n_iter)
            self.write_scalar(
                "train/logits_s_diff",
                (logits_s_prime - logits_s_dprime).pow(2).mean(),
                n_iter,
            )
            self.write_scalar(
                "train/label_dist_kl",
                F.kl_div(label_dist_s.log(), label_dist_t),
                n_iter,
            )

        # Update learning rate scheduler
        self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_s_w = batch_x["img"]
        input_s_s = batch_x["img2"]
        label_s = batch_x["label"]

        input_t_w = batch_u["img"]
        input_t_s = batch_u["img2"]
        # label_t is used only for evaluating pseudo labels' accuracy
        label_t = batch_u["label"]

        input_s_w = input_s_w.to(self.device)
        input_s_s = input_s_s.to(self.device)
        label_s = label_s.to(self.device)

        input_t_w = input_t_w.to(self.device)
        input_t_s = input_t_s.to(self.device)
        label_t = label_t.to(self.device)

        return input_s_w, input_s_s, label_s, input_t_w, input_t_s, label_t

    def _write_train_imgs(self, imgs_dict, n_imgs=16):
        mean = self.cfg.INPUT.PIXEL_MEAN
        std = self.cfg.INPUT.PIXEL_STD

        for t, img in imgs_dict.items():
            self._writer.add_images(
                t,
                unnormalize_img(img[:n_imgs], mean, std),
                0,
            )

    @staticmethod
    def _disable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = False

        model.apply(fn)

    @staticmethod
    def _enable_batchnorm_tracking(model):
        def fn(module):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.track_running_stats = True

        model.apply(fn)

    @staticmethod
    def _compute_loss_target_weight(n_iter, max_iter):
        mu = 0.5 - math.cos(min(math.pi, 2 * math.pi * n_iter / max_iter)) / 2

        return mu
