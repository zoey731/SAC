import torch

ckpt_path = "/home/zhoujie/Transformer-M/logs/L12-old.pt"  
ckpt = torch.load(ckpt_path, map_location="cpu")

print("Top-level keys:")
print(ckpt['model'].keys())
print(len(ckpt['model']))


print("\n--- Extra state ---")
print(ckpt.get("extra_state", None))

print("\n--- Optimizer ---")
print(ckpt.get("optimizer", None))

print("\n--- LR scheduler ---")
print(ckpt.get("lr_scheduler", None))

print("best_loss:", ckpt.get("best_loss", "NOT FOUND"))
