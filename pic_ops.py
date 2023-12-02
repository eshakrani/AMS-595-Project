import cv2         
import time        
from PIL import Image 
import re 

def capture_image(box_size=None):
    """Captures a frame from the device's live webcam feed

    Uses the cv2 library to access the device's webcam and capture one frame 
    from the feed. This frame is captured 5 seconds after the camera window
    opens and saved to the 'captures' directory as an image file titled "capture.jpg".

    If box_size is specified as a 2-dimensional tuple (a, b), then a box of those
    dimensions will be drawn in the capture window. If box_size = None (default), 
    there will be no box.

    The function returns:
        - 0 if an error occurred 
        - 1 if the operation succeeded but no box_size was specified
        - coordinates and size of the box if the operation succeeded and 
             box_size was specified
    """

    # open the webcam
    cap = cv2.VideoCapture(0) 

    # if unable to open the webcam, return None and exit the function
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return None

    # record the start time so we can tell when 5 seconds have passed
    t1 = time.time() 

    # boolean indicator of whether the operation was a success/failure
    flag = False

    while True:
        try:
            ret, frame = cap.read()

            # if unable to capture the frame successfully, exit the function
            if not ret:
                print('Error: Could not capture image.')
                flag = False
                break 

            # 'q' or 'ESC' can be used to escape from the program
            if cv2.waitKey(1) in [ord('q'), 27]:
                print('Escaped program with "q" or "ESC"')
                flag = False
                break

            if box_size:
                # draw a box on the window to show the user where to put their head
                box_color = (255, 255, 255)
                box_thickness = 3

                # specify the coordinates of the top left corner of the box
                x, y = (int((frame.shape[1] - box_size[0]) / 2),
                        int((frame.shape[0] - box_size[1]) / 2))

                cv2.rectangle(frame,
                            (x, y),                               # start coordinates
                            (x + box_size[0], y + box_size[1]),   # end coordinates
                            box_color,
                            box_thickness)
            
            # flip the camera feed and display it on the screen
            cv2.imshow('Face Recognition', cv2.flip(frame, 1))

            # once 5 seconds have passed, capture the user's image
            # and save it to the system
            if time.time() - t1 > 5:
                cap.release()
                cv2.imwrite('captures/capture.jpg', frame)
                flag = True 
                break 

        except KeyboardInterrupt:
            print('Keyboard Interrupt. Exiting program....')

        except Exception as e:
            print(e)
    
    # release the webcam if necessary
    cap.release()
    cv2.destroyWindow('Face Recognition')

    # return value depends on whether the operation succeeded or failed
    # if any issues arose the file wasn't saved -> None
    # if the image was successfully saved:
    #   if there was a box -> return the start coordinates, height, and 
    #     width of the box in case the image needs to be cropped later
    #   if there was no box -> return 1 to show the image was saved
    if not flag:
        return None 
    else:
        return {'x': x, 
                'y': y, 
                'width': box_size[0], 
                'height': box_size[1]} if box_size else 1
    

def crop_image(file_path=None, x=0,  y=0, width=0, height=0):
    """Crops an image to a specific size and saves it as a new file

    Reads in an image file using the specified file_path and crops it
    using the values passed into the function. The cropped image is
    then saved to the current directory under the name 
    "cropped_[file_path]"
    The path to the cropped image is also returned from the function

    x, y: the coordinates of the top-left corner of the cropping rectangle
    width, height: dimensions of the cropping rectangle
    """
    if not file_path:
        print("File path not provided.")
        return
    
    try:
        # extract the 'name' of file "name.jpg"
        name_pattern = re.compile('/(.+)\.')
        name = re.findall(name_pattern, file_path)[0]

        # open the image
        im = Image.open(file_path) 

        # crop the image
        im_c = im.crop((x, y, x + width, y + height))

        # save the adjusted image to the system and return its path
        output_path = f'captures/cropped_{name}.jpg'
        im_c.save(output_path)
        return output_path

    except Exception as e:
        print(e)


def resize_image(file_path=None, output_path=None, width=0, height=0):
    """Resizes a given image to a new width and height

    Reads in an image specified by file_path and adjusts
    its current width and height to the new values denoted 
    by the parameters width and height, respectively

    The resized image is then saved to the current directory
    under the name "resized_[file_path]"

    The path to the resized image is returned from the function
    """
    
    if not file_path:
        print("File path not provided.")
        return 

    try:
        # extract the 'name' of file "name.jpg"
        # name_pattern = re.compile('/(.+)\.')
        # name = re.findall(name_pattern, file_path)[0]

        # open the image
        im = Image.open(file_path)

        # resize the image
        im_r = im.resize((width, height), Image.LANCZOS)

        # save the resized image and return its path
        
        im_r.save(output_path)
        return output_path 

    except Exception as e:
        print(f'Error with resizing {file_path}')
        print(e)

if __name__ == '__main__':
    # capture_image()
    # crop_image()
    # resize_image()
    pass 