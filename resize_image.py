from PIL import Image 

def resize_image(file_path=None, width=0, height=0):
    """Resizes a given image to a new width and height

    Reads in an image specified by file_path and adjusts
    its current width and height to the new values denoted 
    by the parameters width and height, respectively

    The resized image is then saved to the current directory
    under the name "resized_[file_path]"
    """
    
    if not file_path:
        return 

    try:
        # open the image
        im = Image.open(file_path)

        # resize the image
        im_r = im.resize((width, height), Image.LANCZOS)

        # save the resized image
        im_r.save(f'captures/resized_{file_path}')

    except Exception as e:
        print(e)

if __name__ == '__main__':
    resize_image()