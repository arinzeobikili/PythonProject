import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

#print(digits.target.shape)

images_and_labels = list(zip(digits.images, digits.target))
#print(images_and_labels)

for index, (image, label) in enumerate(images_and_labels[:6]):
    plt.subplot(2,3, index+1) #2 rows, and 3 columns
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i' %label)

plt.show()

#to apply a classifier on this data, we need to flatten the image: instead of a 8x8 matrix
#we have to use a one-dimensional array with 64 items
data = digits.images.reshape((len(digits.images), -1))

#print(data)

classifier = svm.SVC(gamma=0.001)

train_test_split = int(len(digits.images)*0.75) #75% to training
classifier.fit(data[:train_test_split], digits.target[:train_test_split])

#predict the value of digit on the 25%
expected = digits.target[train_test_split:]
predicted = classifier.predict(data[train_test_split:])

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print(accuracy_score(expected, predicted))

# let's test on the last few images
plt.imshow(digits.images[-2], cmap=plt.cm.gray_r, interpolation='nearest')
print("Prediction for test image: ", classifier.predict(data[-2].reshape(1, -1)))

plt.show()






