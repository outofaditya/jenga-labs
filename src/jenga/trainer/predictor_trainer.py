import os
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import Trainer
from jenga.models.predictor import PrunableAttnPredictor


def extract_pruned_config(model: nn.Module, is_opt: bool = False) -> Dict[str, Any]:
    # capture per-layer predictor out_features so a saved checkpoint can rebuild the pruned shape
    cfg = {"layers": []}
    if is_opt:
        for layer in model.model.decoder.layers:
            predictor = layer.self_attn.predictor
            if isinstance(predictor, PrunableAttnPredictor):
                layer_cfg = predictor.get_current_outdims()
                cfg["layers"].append(layer_cfg)
            else:
                cfg["layers"].append(None)
    else:
        for layer in model.model.layers:
            predictor = layer.self_attn.predictor
            if isinstance(predictor, PrunableAttnPredictor):
                layer_cfg = predictor.get_current_outdims()  # dict
                cfg["layers"].append(layer_cfg)
            else:
                cfg["layers"].append(None)
    return cfg


class PredictorTrainer(Trainer):
    def __init__(
        self,
        orig_weight_training=False,
        gate_loss_scale=1.0,
        fix_mask_predictor=False,
        use_mse_loss=False,
        is_opt=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if use_mse_loss:
            self.loss_fct = torch.nn.MSELoss()
        else:
            self.loss_fct = torch.nn.SmoothL1Loss()
        self.gate_loss_scale = gate_loss_scale
        self.orig_weight_training = orig_weight_training
        self.fix_mask_predictor = fix_mask_predictor
        self.is_opt = is_opt

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)

        predict_mask = outputs.get("predict_mask")
        pooling_gt = outputs.get("pooling_gt")
        original_loss = outputs.get("loss")

        if not return_outputs:
            del outputs

        mask_loss = 0
        if not self.fix_mask_predictor:
            for pm, gt in zip(predict_mask, pooling_gt):
                mask_loss += self.loss_fct(pm, gt)

        del predict_mask
        del pooling_gt

        if self.orig_weight_training:
            tok_loss = original_loss + self.gate_loss_scale * mask_loss
        else:
            tok_loss = self.gate_loss_scale * mask_loss

        return (tok_loss, outputs) if return_outputs else tok_loss

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        pruned_cfg = extract_pruned_config(self.model, self.is_opt)
        params_to_save = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save(params_to_save, os.path.join(output_dir, "predictor.pth"))
        torch.save(pruned_cfg, os.path.join(output_dir, "pruned_config.pth"))


class DynamicPruningPredictorTrainer(PredictorTrainer):
    """runs prune_neurons() on every PrunableAttnPredictor every prune_interval steps."""

    def __init__(self, prune_interval=100, zero_ratio_threshold=0.8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_interval = prune_interval
        self.zero_ratio_threshold = zero_ratio_threshold

    def compute_loss(self, model, inputs, return_outputs=False):
        loss_tuple = super().compute_loss(model, inputs, return_outputs=True)
        step = self.state.global_step

        if step > 0 and (step % self.prune_interval == 0) and step < 620:
            times = step // self.prune_interval
            thresh = self.zero_ratio_threshold - (times - 1) * 0.05
            for module_name, module in model.named_modules():
                if isinstance(module, PrunableAttnPredictor):
                    module.prune_neurons(
                        step_count_threshold=self.prune_interval,
                        zero_ratio_threshold=thresh,
                    )

        if return_outputs:
            return loss_tuple
        else:
            return loss_tuple[0]
