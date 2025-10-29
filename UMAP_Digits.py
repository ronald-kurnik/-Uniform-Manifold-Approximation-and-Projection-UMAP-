# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 17:56:57 2025

@author: Ron
"""

import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data  # 1797 x 64
y = digits.target

# UMAP reduction
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)
embedding = reducer.fit_transform(X)

# Plot
plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of Digits dataset')
plt.show()