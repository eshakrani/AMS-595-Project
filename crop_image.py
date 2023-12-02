from PIL import Image 

def crop_image(file_path=None, x=0,  y=0, width=0, height=0):
    """Crops an image to a specific size and saves it as a new file

    Reads in an image file using the specified file_path and crops it
    using the values passed into the function. The cropped image is
    then saved to the current directory under the name 
    "cropped_[file_path]"

    x, y: the coordinates of the top-left corner of the cropping rectangle
    width, height: dimensions of the cropping rectangle
    """
    if not file_path:
        return
    
    try:
        name = file_path[file_path.index('/') + 1: file_path.index()]
        # open the image
        im = Image.open(file_path) 

        # crop the image
        im_c = im.crop((x, y, x + width, y + height))

        # save the adjusted image to the system
        output_path = f'captures/cropped_{file_path}'
        print(output_path)
        im_c.save(output_path)

    except Exception as e:
        print(e)

if __name__ == '__main__':
    crop_image()