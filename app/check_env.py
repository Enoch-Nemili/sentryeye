"""Quick environment check for SentryEye — confirms the core libraries import."""

import numpy as np

print("--- SentryEye environment check ---")
print("NumPy       :", np.__version__)

try:
    import cv2
    print("OpenCV      :", cv2.__version__)
except Exception as e:
    print("OpenCV      : NOT available ->", e)

try:
    import torch
    print("PyTorch     :", torch.__version__)
    print("Apple GPU (MPS) available:", torch.backends.mps.is_available())
except Exception as e:
    print("PyTorch     : NOT available ->", e)

try:
    import ultralytics
    print("Ultralytics :", ultralytics.__version__)
except Exception as e:
    print("Ultralytics : NOT available ->", e)

print("\nIf all four printed versions, the environment is ready. ✅")
