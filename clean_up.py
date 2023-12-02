import os 

def clean_up(dir_path=None):
    """Deletes the files located in a specific directory

       Searches for the directory specified by dir_path and
       deletes all files and folders found at that location
    """

    try:
        # check if dir_path leads to a valid directory
        if not os.path.isdir(dir_path):
            raise ValueError(f'{dir_path} is not a valid directory.')
        
        print(f'Emptying directory: {dir_path}')
        for name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, name)

            try:
                # check if file_path is a file
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f'Removed file: {file_path}')
                
                # check if file_path is a directory
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
                    print(f'Removed directory: {file_path}')

            except Exception as e:
                print(e)

    except ValueError as v:
        print(f'Error: {v}')

if __name__ == '__main__':
    clean_up('gd_results')
    pass 
