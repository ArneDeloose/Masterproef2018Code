import numpy as np

X = np.array([[5,3],  
    [10,15],
    [15,12],
    [24,10],
    [30,30],
    [85,70],
    [71,80],
    [60,78],
    [70,55],
    [80,91],])

import matplotlib.pyplot as plt

labels = range(1, 11)  
plt.figure(figsize=(10, 7))  
plt.subplots_adjust(bottom=0.1)  
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):  
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()  

from scipy.cluster.hierarchy import dendrogram, linkage  
from matplotlib import pyplot as plt

linked = linkage(X, 'single')

labelList = range(1, 11)

#plt.figure(figsize=(10, 7))  
dendrogram(linked)

label_colors = {'0': '#B061FF','1': '#B061FF', '2': '#B061FF', '3': '#B061FF',
                '4': '#B061FF', '5': '#B061FF','6': '#B061FF','7': '#B061FF',
                '8': '#B061FF','9': '#B061FF'}

ax = plt.gca()
xlbls = ax.get_xmajorticklabels()
for lbl in xlbls:
    lbl.set_color(label_colors[lbl.get_text()])

plt.show()


# see question for code prior to "color mapping"

# Color mapping
#dflt_col = "#808080"   # Unclustered gray
D_leaf_colors = {1: dflt_col,
                 2: "#B061FF", # Cluster 1 indigo
                 3: "#B061FF",
                 4: "#B061FF",
                 5: "#B061FF",
                 6: "#B061FF",
                 7: "#B061FF",
                 8: "#B061FF", # Cluster 2 cyan
                 9: "#B061FF",
                 10: "#B061FF",
                 11: "#61ffff",
                 12: "#61ffff",
                 13: "#61ffff",
                 14: "#61ffff",
                 15: "#B061FF",
                 16: "#61ffff",
                 17: "#61ffff",
                 18: "#61ffff",
                 19: "#61ffff",
                 20: "#61ffff",
                 21: "#61ffff",
                 22: "#61ffff",
                 23: "#61ffff",
                 24: "#61ffff",
                 25: "#61ffff",
                 26: "#61ffff",
                 27: "#61ffff",
                 28: "#61ffff",
                 29: "#61ffff"
                 }

# Dendrogram
#D = dendrogram(Z=Z, labels=DF_dism.index, color_threshold=None,
  #leaf_font_size=12, leaf_rotation=45, link_color_func=lambda x: link_cols[x])