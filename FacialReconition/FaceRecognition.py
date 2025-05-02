# Import essential libraries for data manipulation, scientific computing, and visualization
import matplotlib.pyplot as plt  # For plotting data
import numpy as np  # For numerical computations

# Import libraries for working with datasets, machine learning models, and evaluation
from sklearn.datasets import fetch_olivetti_faces  # To load the Olivetti Faces dataset
from sklearn.decomposition import PCA  # For dimensionality reduction with Principal Component Analysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, \
    cross_val_score  # To split data into training and testing sets
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load the Olivetti Faces dataset
# This dataset contains 400 grayscale images of 40 people (10 images per individual)
olivetti_data = fetch_olivetti_faces()

# Extract image data (features) and corresponding labels (targets)
features = olivetti_data.data  # Each row corresponds to a flattened image
targets = olivetti_data.target  # Target labels represent individual identities

fig, subplots = plt.subplots(nrows=5, ncols=8, figsize=(14, 8))
subplots = subplots.flatten()

# Loop through each unique user ID and display one sample image for that user
for unique_user_id in np.unique(targets):
    image_index = unique_user_id * 8
    subplots[unique_user_id].imshow(features[image_index].reshape(64, 64), cmap='gray')
    subplots[unique_user_id].set_xticks([])
    subplots[unique_user_id].set_yticks([])
    subplots[unique_user_id].set_title("Face id: %s"%unique_user_id)

plt.suptitle("Olivetti Faces Dataset - 40 people")
plt.show()


#Show images of the first person (faceid = 0)
fig, subplots = plt.subplots(nrows=1, ncols=10, figsize=(18, 9))

# Display all images (10 total) of the first person (Face ID = 0) in a single row
for j in range(10):
    subplots[j].imshow(features[j].reshape(64, 64), cmap='gray')
    subplots[j].set_xticks([])
    subplots[j].set_yticks([])
    subplots[j].set_title("Face id = 0")
plt.show()

# Split dataset into training (75%) and test (25%) sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=42)


# Reduce the dimensionality of the features while maintaining key variance using PCA
pca = PCA(n_components=100, whiten=True)  # Keep 100 principal components and apply whitening for decorrelation
X_train_pca = pca.fit_transform(X_train)  # Fit PCA on the training data and transform it
X_test_pca = pca.transform(X_test)  # Apply the trained PCA transformation to the test data

# After performing PCA, calculate the number of eigenfaces (principal components)
number_of_eigenfaces = len(pca.components_)  # In this case, it's 100 as set earlier
eigenfaces = pca.components_.reshape((number_of_eigenfaces, 64, 64))

# Create a grid to display eigenfaces
fig, subplots = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))  # 10x10 grid for 100 eigenfaces
subplots = subplots.flatten()  # Flatten 2D array of subplots for easy iteration

# Loop through each eigenface and plot it on a subplot
for i in range(number_of_eigenfaces):
    subplots[i].imshow(eigenfaces[i], cmap='gray')  # Display the eigenface as a grayscale image
    subplots[i].set_xticks([])  # Remove x-axis ticks for better visualization
    subplots[i].set_yticks([])  # Remove y-axis ticks for better visualization

# Set the title of the figure
plt.suptitle("Eigenfaces of the Olivetti Faces Dataset")  # Add a title for the eigenfaces grid
# Show the plotted figure
plt.show()

# Iterate through various machine learning models and evaluate their accuracy
# on the Olivetti Faces Dataset using data projected through PCA.
models = [("Logistic Regression", LogisticRegression()),
          ("Support Vector Machine", SVC()),
          ("Naive Bayes", GaussianNB())]

for name, model in models:
    classifier = model.fit(X_train_pca, y_train)
    y_pred = classifier.predict(X_test_pca)
    print("Model: %s, Accuracy: %.2f%%" % (name, 100 * classifier.score(X_test_pca, y_test)))


# Perform 5-fold cross-validation for each machine learning model in the models list.
# This evaluates the model's accuracy and standard deviation across different splits of the training data.
    for name, model in models:
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)  # Initialize K-Fold cross-validator
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=kfold)  # Compute cross-validation scores
        print("Model: %s, Accuracy: %.2f%% (std: %.2f%%)" % (name, cv_scores.mean() * 100, cv_scores.std() * 100))
