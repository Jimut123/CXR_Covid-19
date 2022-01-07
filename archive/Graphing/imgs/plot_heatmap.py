import numpy as np
import matplotlib.pyplot as plt

heat_map_save_name = "pneu_423_heatmap.npy"
heat_map = np.load(heat_map_save_name)
plt.imshow(heat_map,cmap='gray')
plt.show()






