import os
from configuration import Config
import shutil



def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
path_dir = Config.pascal_voc_root + Config.work_type
createFolder(path_dir+"/Annotations")
createFolder(path_dir+"/JPEGImages")

file_list = os.listdir(path_dir)
print(file_list)

for file in file_list:
    if file == "Annotations" or file == "JPEGImages":
        continue
    source = path_dir+"/" + file

    if ".jpg" ==source[-4:]:
        destination = path_dir+"/JPEGImages/"+ file
    elif ".xml" == source[-4:]:
        destination = path_dir+"/Annotations/"+ file
    shutil.move(source,destination)