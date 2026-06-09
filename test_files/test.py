import json
import os
import subprocess
import random
import matplotlib.pyplot as plt

json_path = "segments_rms.json"

with open(json_path, 'r') as f:
    segments = json.load(f)


times = [s["start"] for s in segments]
rms   = [s["rms"]   for s in segments]

plt.figure(figsize=(20, 4))
plt.plot(times, rms, linewidth=0.5)
plt.axhline(y=0.06, color='red', linestyle='--', label='threshold')
plt.xlabel("Time (seconds)")
plt.ylabel("RMS energy")
plt.title("Audio excitement over match")
plt.legend()
plt.savefig("rms_plot.png", dpi=150)
print("Plot successfully saved to rms_plot.png. You can open this file in VS Code to view it!")
