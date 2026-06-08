import sys
print("start", flush=True)
import numpy as np
print("numpy ok", flush=True)
import matplotlib
print("mpl backend:", matplotlib.get_backend(), flush=True)
import matplotlib.pyplot as plt
print("pyplot ok", flush=True)
fig = plt.figure(figsize=(4, 3))
print("figure ok", flush=True)
ax = fig.add_subplot(111)
print("axes ok", flush=True)
ax.plot([1, 2, 3], [1.0, 0.5, 0.1])
print("plot ok", flush=True)
fig.savefig("test_out.png", dpi=72)
print("save ok", flush=True)
plt.close("all")
print("close ok", flush=True)
sys.exit(0)
