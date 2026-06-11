import argparse
import os

import torch

from jenga.models.predictor import PrunableAttnPredictor, PrunableAttnPredictorInfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictor_path",
        type=str,
        default="logs/ablations/predictor/llama2_16384_predictor_red",
    )
    args = parser.parse_args()

    pruned_cfg = torch.load(os.path.join(args.predictor_path, "pruned_config.pth"))
    layers_cfg = pruned_cfg["layers"]
    predictor_ori = PrunableAttnPredictor(dim=256, hidden_dim=64, n_head=32)
    params_ori = sum(p.numel() for p in predictor_ori.parameters())

    total_params = 0
    for layer, layer_cfg in enumerate(layers_cfg):
        predictor = PrunableAttnPredictorInfer(
            dim=256,
            hidden_dim=64,
            q1_outdim=layer_cfg["q1_outdim"],
            q2_outdim=layer_cfg["q2_outdim"],
            k1_outdim=layer_cfg["k1_outdim"],
            k2_outdim=layer_cfg["k2_outdim"],
        )
        params = sum(p.numel() for p in predictor.parameters())
        print(
            f"Layer {layer}, Pruned params: {params}, original params: {params_ori}, Pct : {params / params_ori:.2f}"
        )
        total_params += params

    total_memory_mb = total_params * 2 / (1024**2)
    print(f"Total memory: {total_memory_mb:.2f} MB")
