from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
#print(digits)

X_digits = digits.data
Y_digits = digits.target

print(X_digits.shape)

estimator = PCA(n_components=10) #transform 64 features into x components. Part of PCA analysis of reducing features
X_pca = estimator.fit_transform(X_digits)

print(X_pca.shape)

colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

for i in range(len(colors)):
    px = X_pca[:, 0][Y_digits == i]
    py = X_pca[:, 1][Y_digits == i]
    plt.scatter(px, py, c=colors[i])
    plt.legend(digits.target_names)


# explained variance shows how much information can be attributed to the principal components
print("Explained variance: %s" % estimator.explained_variance_ratio_)


plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()