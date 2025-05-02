import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import datasets
import numpy as np

X, y = datasets.make_moons(n_samples=1500, noise=.05)
#print(X.shape)

#seperate features by column
x1 = X[:, 0]
x2 = X[:, 1]

plt.scatter(x1, x2, s=5)
plt.show()

dbscan = DBSCAN(eps=0.3) #max distance from one point to be considered in the same cluster
dbscan.fit(X)

y_pred= dbscan.labels_.astype(int) #np.int is deprecated
print(y_pred)

colors = np.array(['#ff0000', '#00ff00'])
plt.scatter(x1, x2, s=5, color=colors[y_pred])
plt.show()

#try kmeans clustering. You'll see some mislabelled or misclassified items in the same cluster
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_pred = kmeans.labels_.astype(int)

plt.scatter(x1, x2, s=5, color=colors[y_pred])
plt.show()
