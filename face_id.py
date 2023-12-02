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


# if the files containing the tuned weight and bias parameters
# both exist, that means the model has already been trained
w_path = 'gd_results/training_weights3.mat'
b_path = 'gd_results/training_biases3.mat'

trained = True

for path in [w_path, b_path]:
    if not os.path.exists(path):
        trained = False 
        break 

image_path = 'lfw_data'
image_size = (64, 64)

target_name = 'George_W_Bush'

# if either file does not exist, the model needs to be trained
if not trained:
    '''
    For this specific case, we are training the model to 
    recognize George W. Bush
    '''

    print('Did not find existing weight and bias values.')

    # load the image set and store it in a list
    # the images will need to be resized to make
    #   computation times reasonable
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

    random.seed(595)
    others_indices = random.sample(others_indices, len(target_indices))

    subset_indices = target_indices + others_indices 
    X_subset = X[subset_indices]
    Y_subset = Y[subset_indices]

    # # separately split both lists into training and testing sets,
    # #   - combine both training sets into one set
    # #   - combine both testing sets into one set
    # # this ensures that the overall training and testing sets will 
    # #   have proportional amounts of target samples instead of 
    # #   randomly splitting the entire dataset 
    # target_ind_train, target_ind_test = train_test_split(target_indices, 
    #                                                      test_size=0.25, 
    #                                                      random_state=595)
    # others_ind_train, others_ind_test = train_test_split(others_indices,
    #                                                      test_size=0.25,
    #                                                      random_state=595)

    # train_indices = target_ind_train + others_ind_train 
    # test_indices  = target_ind_test  + others_ind_test

    # # use these lists to extract the actual training/testing data
    # X_train = X[train_indices]
    # X_test

    # train a model on the dataset and get the tuned parameters
    # params = lr.train(X, Y, w_path, b_path)
    params = lr.train(X_subset, Y_subset, w_path, b_path)
    w = params['w']
    b = params['b']

# use the device's camera to take a picture of the user
# provide a box in the camera feed for the user to position 
#   their head
capture = capture_image(box_size=(750, 750))

# if capture_image() returns None, then there was a problem
#  getting the image from the camera. Exit the program.
if not capture:
    print('Error capturing picture.')
    exit(1)

# since box_size was specified as a parameter for the 
#   capture_image() function, the returned value gives
#   the coordinates and size of the box
box_x = capture['x']
box_y = capture['y']
box_width = capture['width']
box_height = capture['height']

# to match the shape of the training data, the capture will
#   be cropped so that only the selection in the box remains
cropped_path = crop_image(file_path='captures/capture.jpg',
                          x=box_x,
                          y=box_y,
                          width=box_width,
                          height=box_height)

# now the captured image is the same shape as the training images,
#  but it has to be resized from (750x750) to match their (64x64) size 
resized_path = resize_image(file_path=cropped_path,
                            output_path='captures/resized_capture.jpg',
                            width=image_size[0],
                            height=image_size[1])


# store the resized image of the user's face
# face = cv2.imread(resized_path)
face = cv2.imread('lfw_data/George_W_Bush/George_W_Bush_0001.jpg')
face = cv2.resize(face, image_size)
print(face)
# using the weights and biases previously tuned by the training
#   process, pass the face image into the classifier and get
#   the predicted label as well as its probability
class_results = lr.classification(face, w_path, b_path)
face_label = class_results['label']
face_prob  = class_results['prob']

# if the returned label is 1 with a probability of at least 80%, 
#   we will say that the face in the image was correctly recognized 
#   as the target person
# if either of the following conditions is true, then we say that 
#   the face in the image is not the target person
#       1. predicted label is 0
#       2. probability is less than 80%
#   in this case, prompt the user for a password to prove
#     their identity
if face_label == 1 and face_prob >= 0.8:
    print(f'Face recognized as {target_name}.')
    print('Unlocking...')

else:
    print('Face not recognized.')
    pwd = input('Please enter your password: ')
    if pwd == creds.gwb_password:
        print('Unlocking...')
    else:
        print('Incorrect password.')
        print('Exiting program.')
        exit(2)

# delete any files that were created
clean_up('captures')

    
