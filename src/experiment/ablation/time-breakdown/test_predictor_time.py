import argparse
import os

import torch

from jenga.models.predictor import PrunableAttnPredictorInfer

WARMUP_RUNS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=16384)
    parser.add_argument("--predictor_path", type=str, default="checkpoints/predictor")
    args = parser.parse_args()

    pruned_cfg = torch.load(os.path.join(args.predictor_path, "pruned_config.pth"))
    times = []
    for layer_cfg in pruned_cfg["layers"]:
        predictor = (
            PrunableAttnPredictorInfer(
                dim=128,
                hidden_dim=512,
                q1_outdim=layer_cfg["q1_outdim"],
                q2_outdim=layer_cfg["q2_outdim"],
                k1_outdim=layer_cfg["k1_outdim"],
                k2_outdim=layer_cfg["k2_outdim"],
            )
            .to(torch.bfloat16)
            .to("cuda")
            .eval()
        )

        with torch.no_grad():
            inp = torch.randn(
                1, args.seq_len, 4096, dtype=torch.bfloat16, device="cuda"
            )
            for _ in range(WARMUP_RUNS):
                predictor(inp)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            predictor(inp)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

    print(f"seq_len: {args.seq_len} time: {sum(times)} ms")
