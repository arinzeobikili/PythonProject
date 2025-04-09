from mlxtend.data import mnist_data
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

mnist_data = fetch_openml('mnist_784')

features = mnist_data.data
target = mnist_data.target

print(features.shape) #70,000 by 748 - a lot of features

train_img, test_img, train_lbl, test_lbl = train_test_split(features, target, test_size=0.15, random_state=0)

scaler = StandardScaler()
scaler.fit(train_img) #calculate mean and standard deviation on train_img
train_img = scaler.transform(train_img) #apply z transformation to train_img
test_img = scaler.transform(test_img) #apply z transformation to test_img

print("Training image shape before applying PCA: ",train_img.shape)

# we keep 95% variance - so that's 95% of the original information - different from defining number of components in PCA1
pca = PCA(.95)
pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

print("Training image shape after applying PCA: ",train_img.shape)