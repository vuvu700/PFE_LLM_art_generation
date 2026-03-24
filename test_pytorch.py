import time, sys

t0 = time.perf_counter()
print(f"started version: {sys.version_info[: 3]}")

import torch

print(f"imported torch in {time.perf_counter()-t0}")
