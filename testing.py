import logistic_regression as lr 
from pic_ops import capture_image, crop_image, resize_image
from clean_up import clean_up
from load_images import load_images_from_folder, load_image_names_from_folder
import os 
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 
import credentials as creds
import random
import scipy


# if the files containing the tuned weight and bias parameters
# both exist, that means the model has already been trained
w_path = 'gd_results/training_weights3.mat'
b_path = 'gd_results/training_biases3.mat'

trained = True 

image_path = 'lfw_data'
image_size = (100, 100)

target_name = 'George_W_Bush'

images = load_images_from_folder(image_path, size=image_size)

# convert the list of images into a numpy array
X = np.stack(images, axis=0)

# flatten and normalize the data
X = X.reshape((X.shape[0], np.prod(X.shape[1:4])))
X = X / 255 

print(f'Shape of data: {X.shape}')

# get a list of each image's corresponding name
names = load_image_names_from_folder(image_path)

# convert the names into labels:
# target name   --> 1
# otherwise     --> 0
Y = np.array([1 if name == target_name else 0 for name in names])
Y = Y.reshape((Y.shape[0], 1))
print(f'Shape of labels: {Y.shape}')
# # extract the indices corresponding to the target name
target_indices = [i for i in range(len(names)) if target_name in names[i]]
others_indices = [i for i in range(len(names)) if i not in target_indices]
test_indices = [i for i in range(len(names)) if i not in others_indices]

random.seed(595)
others_indices = random.sample(others_indices, len(target_indices))
test_indices = random.sample(test_indices, len(target_indices))

subset_indices = target_indices + others_indices 
X_subset = X[subset_indices]
Y_subset = Y[subset_indices]

test_subset_indices = target_indices + test_indices 
X_test_subset = X[test_subset_indices]
Y_test_subset = Y[test_subset_indices]


w = scipy.io.loadmat(w_path)['weights']
b = scipy.io.loadmat(b_path)['biases']

#sigmoid with final w and b, fitted to binary 1 or 0 for train data
A_train = lr.sigmoid(np.dot(X_subset, w) + b)
train_predictions = (A_train >= .5).astype(int)

#finf accuracy of predictions and print
train_accuracy = np.mean(train_predictions == Y_subset)

print(f"Train Set Accuracy: {train_accuracy:.2f}")

#sigmoid with final w and b, fitted to binary 1 or 0 for test data
A_test = lr.sigmoid(np.dot(X_test_subset, w) + b)
test_predictions = (A_test >= .5).astype(int)

#finf accuracy of predictions and print
test_accuracy = np.mean(test_predictions == Y_test_subset)

print(f"Test Set Accuracy: {test_accuracy:.2f}")