import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#find pattern based on k-means. Unsupervised learning

#random values on a scatter plot
x, y = make_blobs(n_samples=100, centers=5, random_state=0, cluster_std=.3)

#print(y)
plt.scatter(x[:, 0], x[:, 1], s=50)

#create model based on 3 clusters
model = KMeans(5)
model.fit(x)
y_kmeans = model.predict(x)
print(y_kmeans)


plt.scatter(x[:,0], x[:,1], c=y_kmeans, s=50, cmap='rainbow') #colors for each integer in cluster

plt.show()