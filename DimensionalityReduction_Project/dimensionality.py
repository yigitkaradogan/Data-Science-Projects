import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
from os import path
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.manifold import Isomap, TSNE

ROOT_DIR = path.abspath(os.curdir)
DATA_DIR = path.join(ROOT_DIR, 'dimensionality')

np.random.seed(1)

fashion_data = np.loadtxt(path.join(DATA_DIR, 'fashion_mnist_data.txt'))
fashion_data = (fashion_data * 255).astype(np.uint8)

labels_data = np.loadtxt(path.join(DATA_DIR, 'fashion_mnist_labels.txt'), dtype=np.int32)
labels_data = labels_data.reshape((10000, 1))
data_full = np.concatenate((fashion_data, labels_data), axis=1)

labels_data = labels_data.reshape((10000, ))
fashion_data = fashion_data.reshape((10000, 28, 28))
groups = [fashion_data[labels_data == label] for label in np.unique(labels_data)]

x_train = []
x_test = []
y_train = []
y_test = []

for label in np.unique(labels_data):
    data = fashion_data[labels_data == label]
    data_train, data_test = train_test_split(
        data, test_size=0.5, random_state=1
    )

    x_train.append(data_train)
    x_test.append(data_test)
    y_train.extend([label] * len(data_train))
    y_test.extend([label] * len(data_test))

x_train = np.vstack(x_train)
y_train = np.array(y_train)
x_test = np.vstack(x_test)
y_test = np.array(y_test)

#Question 1.1
mean_data = np.mean(np.concatenate((x_train, x_test), axis=0), axis=0)

x_train_centered = x_train - mean_data
x_test_centered = x_test - mean_data

x_train_centered = x_train_centered.reshape(x_train_centered.shape[0], -1)
x_test_centered = x_test_centered.reshape(x_test_centered.shape[0], -1)
x_centered = np.concatenate((x_train_centered, x_test_centered), axis=0)

mean_data = mean_data.reshape(784,)
data_full = data_full[:, 1:]

centered_full_data = data_full - mean_data
centered_full_data = centered_full_data.reshape(10000, 784)

#Question 1.2
pca = PCA()
pca.fit(x_train_centered)

def calculate_cumulative_variance(eigenvalues, threshold):
    cumulative_ratios = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    return np.argmax(cumulative_ratios >= threshold) + 1

threshold = 0.82
threshold_index = calculate_cumulative_variance(pca.explained_variance_, threshold)
print(f"The minimum number of principal components to ensure {threshold*100}% variance: {threshold_index}")

plt.figure(figsize=(7, 6))
plt.plot(pca.explained_variance_, label='Eigenvalues')
plt.fill_between(range(threshold_index), pca.explained_variance_[:threshold_index], alpha=0.4, step='pre')
plt.axvline(x=threshold_index, color='g', linestyle='--', label='Threshold Index')
plt.xlim(-5, 784)
plt.text(threshold_index, 0.05, str(threshold_index), color='g', ha='left', size = 'x-large')
plt.xlabel('Number of Principal Components')
plt.ylabel('Eigenvalue')
plt.legend()
plt.title('Eigenvalues in Descending Order')

plt.figure(figsize=(7, 6))
plt.plot(np.arange(len(pca.explained_variance_)), np.cumsum(pca.explained_variance_)/np.sum(pca.explained_variance_))
plt.xlim(-5, 784)
plt.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold*100}% threshold')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.legend()
plt.title('Cumulative Explained Variance Ratio')

plt.tight_layout()
plt.show()

#Question 1.3
train_mean = np.mean(x_train_centered, axis=0)

plt.figure(figsize=(4, 4))
plt.axis('off')
plt.title('Mean Image of the Training Data')
data = train_mean.reshape((28, 28)).T
plt.imshow(data, cmap='plasma', interpolation='nearest')
plt.show()

fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(10, 8))  # Increased figsize for better visibility
for index in range(30):
    row = index // 5
    col = index % 5
    ax[row, col].set_axis_off()
    component = pca.components_[index, :].reshape((28, 28)).T
    ax[row, col].imshow(component, cmap='plasma', interpolation='nearest')
plt.suptitle('The First 30 Principal Components', fontsize=16)
plt.tight_layout()
plt.show()

# Question 1.4
k_components = [10 * k for k in range(1, 41)]
train_errors = []
test_errors = []

for k in tqdm(k_components):
    # PCA
    pcaK = PCA(n_components=k)
    train_transformed = pcaK.fit_transform(x_train_centered)
    test_transformed = pcaK.transform(x_test_centered)

    # QDA
    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(train_transformed, y_train)
    train_preds = gaussianK.predict(train_transformed)
    test_preds = gaussianK.predict(test_transformed)

    # Calculating errors
    trainErrK = 1 - accuracy_score(y_train, train_preds)
    testErrK = 1 - accuracy_score(y_test, test_preds)

    train_errors.append(trainErrK)
    test_errors.append(testErrK)

plt.figure(figsize=(12, 8))
plt.plot(k_components, train_errors, label='Train Classification Error', color='red')
plt.xlabel('Number of First Principal Components')
plt.ylabel('Train Classification Error')
plt.xticks(k_components)
plt.legend()
plt.title('Train Classification Error for the Quadratic Gaussian Model')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(k_components, test_errors, label='Test Classification Error', color='red')
plt.xlabel('Number of First Principal Components')
plt.ylabel('Test Classification Error')
plt.xticks(k_components)
plt.legend()
plt.title('Test Classification Error for the Quadratic Gaussian Model')
plt.grid(True)
plt.show()

# Question 2
def generate_random_projection_matrix(original_dimensions, new_dimensions):
    projection_matrix = np.random.randn(original_dimensions, new_dimensions)
    return projection_matrix

train_errors_2 = []
test_errors_2 = []
for k in tqdm(k_components):
    projection_matrix = generate_random_projection_matrix(x_train_centered.shape[1], k)

    train_transformed2 = x_train_centered @ projection_matrix
    test_transformed2 = x_test_centered @ projection_matrix

    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(train_transformed2, y_train)
    train_preds2 = gaussianK.predict(train_transformed2)
    test_preds2 = gaussianK.predict(test_transformed2)

    trainErrK2 = 1 - accuracy_score(y_train, train_preds2)
    testErrK2 = 1 - accuracy_score(y_train, test_preds2)

    train_errors_2.append(trainErrK2)
    test_errors_2.append(testErrK2)

plt.figure(figsize=(12, 8))
plt.plot(k_components, train_errors_2, label='Train Classification Error', color='red')
plt.xlabel('Number of First Principal Components')
plt.ylabel('Train Classification Error')
plt.xticks(k_components)
plt.legend()
plt.title('Train Classification Error for the Quadratic Gaussian Model')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(k_components, test_errors_2, label='Test Classification Error', color='red')
plt.xlabel('Number of First Principal Components')
plt.ylabel('Test Classification Error')
plt.xticks(k_components)
plt.legend()
plt.title('Test Classification Error for the Quadratic Gaussian Model')
plt.grid(True)
plt.show()

# Question 3
train_error = []
test_error = []
k_components = [16 * k for k in range(1, 26)]

for k in tqdm(k_components):
    iso = Isomap(n_components=k)
    iso.fit(x_centered, fashion_data)
    iso_transformed_train = iso.transform(x_train_centered)
    iso_transformed_test = iso.transform(x_test_centered)

    gaussianK = QuadraticDiscriminantAnalysis()
    gaussianK.fit(iso_transformed_train, y_train)
    train_preds = gaussianK.predict(iso_transformed_train)
    test_preds = gaussianK.predict(iso_transformed_test)

    trainErrK = 1 - accuracy_score(y_train, train_preds)
    testErrK = 1 - accuracy_score(y_test, test_preds)

    train_error.append(trainErrK)
    test_error.append(testErrK)

min_test_error = min(test_error)
print(f"Minimum test error: {min_test_error}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 24))

# Plot train error
ax1.plot(k_components, train_error, label='Train Classification Error')
ax1.set_xlabel('Number Of Dimensions in Isomap')
ax1.set_ylabel('Train Classification Error')
ax1.set_xticks(k_components)
ax1.legend()
ax1.set_title('Train Classification Error For The Quadratic Gaussian Model')

# Plot test error
ax2.plot(k_components, test_error, label='Test Classification Error', color='green')
ax2.set_xlabel('Number Of Dimensions in Isomap')
ax2.set_ylabel('Test Classification Error')
ax2.set_xticks(k_components)
ax2.legend()
ax2.set_title('Test Classification Error For The Quadratic Gaussian Model')

plt.tight_layout()
plt.show()

# Question 4
tsne = TSNE(n_components=2)
Embedded_data = tsne.fit_transform(centered_full_data)

df = pd.DataFrame(data=Embedded_data, columns=['First Dimension', 'Second Dimension'])
df['Labels'] = labels_data

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x="First Dimension", y="Second Dimension",
    hue="Labels",
    palette=sns.color_palette("hls", 10),
    data=df,
    alpha=0.7,
    legend="full"
)
plt.title('2D Visualization of the Data')
plt.show()