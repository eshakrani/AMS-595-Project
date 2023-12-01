import cv2          # computer vision for webcam
import time         # for measuring elapsed time

def capture_image():
    """Captures a frame from the device's live webcam feed

    Uses the cv2 library to access the device's webcam and capture one frame 
    from the feed. This frame is captured 5 seconds after the camera window
    opens and saved to the current directory as an image file titled "capture.jpg". 
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

            # flip the camera feed and display it on the screen
            cv2.imshow('Face Recognition', cv2.flip(frame, 1))

            # 'q' or 'ESC' can be used to escape from the program
            if cv2.waitKey(1) in [ord('q'), 27]:
                print('Escaped program with "q" or "ESC"')
                flag = False
                break

            # once 5 seconds have passed, capture the user's image
            # and save it to the system
            if time.time() - t1 > 5:
                cv2.imwrite('capture.jpg', frame)
                flag = True 
                break 

        except KeyboardInterrupt:
            print('Keyboard Interrupt. Exiting program....')

        except Exception as e:
            print(e)
    
    # release the webcam if necessary
    cap.release()

    # return value is 1 if the image was successfully saved and None if
    # there were any issues during the process
    return 1 if flag else None 


if __name__ == '__main__':
    capture_image()