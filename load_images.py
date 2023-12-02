import os
import cv2
from pic_ops import resize_image
from tqdm import tqdm 

def load_images_from_folder(folder, size=None):
    """Loads a set of images from a given directory

    Searches for the directory specified by folder and
    loads all the '.jpg' files found in the folder into 
    a python list. 

    If the images need to be resized, the function 
    accepts a tuple in the form (width, height) as 
    size parameter. These values represent the 
    dimensions of the resized image.
    """
    
    # create a list to hold the data for each image
    images = []

    # get an alphabetically sorted list of the folders to iterate through
    folderlist = sorted(os.listdir(folder), key=str.casefold)

    # filter out any file folders with '.' in the name (ex. .DS_store)
    folderlist = list(filter(lambda f: '.' not in f, folderlist))

    try:
        print(f'Reading folder: {folder}')
        # iterate through the list of directories in folder
        for foldername in tqdm(folderlist):

            # access the person-specific folders by appending
            # the current folder's name to the overall folder's name
            folderpath = os.path.join(folder, foldername)

            # if the current path is not a valid directory, don't process it
            if not os.path.isdir(folderpath):
                print(f'{folderpath} is not a valid directory.')
                continue

            # iterate through the list of files in the 
            # person-specific folder
            for filename in os.listdir(folderpath):

                # determine the filepath by appending the file's
                # name to the encompassing folder's path
                filepath = os.path.join(folderpath, filename)

                # if the current path is not a valid file, don't process it
                if not os.path.isfile(filepath):
                    print(f'{filepath} is not a valid file.')
                    continue 
                
                img = cv2.imread(filepath)

                # if the size paramater has a value, resize the image 
                # to those dimensions
                if size:
                    w, h = size 
                    img = cv2.resize(img, (w, h))

                images.append(img)

        return images
            
    # handle any errors that occur during the process
    except Exception as e:
        print(f'Error loading {folder}')
        print(e)

def load_image_names_from_folder(folder):
    """Extracts a person's name from the names of files in a given folder

    Iterates through the directory specified by folder and makes a list
    of people's names found in the names of each file in the directory.
    """

    # make a list to hold the names found
    names = []

    # get an alphabetically sorted list of the folders to iterate through
    folderlist = sorted(os.listdir(folder), key=str.casefold)

    # filter out any file folders with '.' in the name (ex. .DS_store)
    folderlist = list(filter(lambda f: '.' not in f, folderlist))

    try:

        # iterate through the main directory to find the folders named 
        # after people
        for foldername in folderlist:

            # each person-named folder contains images of that person
            # for each folder, iterate through the image files and 
            #   extract the human name from the complete file path
            folderpath = os.path.join(folder, foldername)
            for filename in os.listdir(folderpath):
                name = filename[:filename.rfind("_")]
                names.append(name)
        
        return names
    
    # handle any errors that occur during the process
    except Exception as e:
        print(e)