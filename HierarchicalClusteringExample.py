import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram # hierarchical clustering is in scipy
import matplotlib.pyplot as plt

# Create a 2D array of points (dataset for clustering)
x = np.array([[1, 1], [1.5, 1], [3, 3], [4, 4], [3, 3.5], [3.5, 4]])

# Scatter plot: Visualize the dataset in a 2D space
plt.scatter(x[:, 0], x[:, 1], s=50)
plt.show()

# Compute the linkage matrix using single linkage (minimum distance between clusters)
#produces the linkage distance and dendogram values for the 2-d array x
linkage_matrix = linkage(x, "single")

# Print the linkage matrix to understand how clusters are merged
# The columns are as follows:
# 1. Indices of the clusters being merged
# 2. Distance between the merged clusters
# 3. Number of points in the new cluster
print(linkage_matrix)

# Generate the dendrogram to visualize the hierarchical clustering
# truncate_mode='none' ensures that the entire dendrogram is displayed
dendogram = dendrogram(linkage_matrix, truncate_mode='none')

plt.title("Hierarchical Clustering")
plt.show()
