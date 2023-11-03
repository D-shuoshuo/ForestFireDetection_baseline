from pathlib import Path
from PIL import Image
import imghdr
import os
import cv2

data_root = os.getcwd()
# data_root = os.path.abspath(os.path.join(os.getcwd(), '..')) 
image_path = os.path.join(data_root, "Dataset", "FireNet_dataset")  
train_data_path = os.path.join(image_path, "train") 

def check_images( s_dir, ext_list):
    bad_images=[]
    bad_ext=[]
    s_list= os.listdir(s_dir)
    for klass in s_list:
        klass_path=os.path.join (s_dir, klass)
        print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            for f in file_list:               
                f_path=os.path.join (klass_path,f)
                tip = imghdr.what(f_path)
                if ext_list.count(tip) == 0:
                  bad_images.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img=cv2.imread(f_path)
                        shape=img.shape
                    except:
                        print('file ', f_path, ' is not a valid image file')
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext

source_dir = train_data_path
good_exts=['jpg', 'png', 'jpeg', 'gif', 'bmp' ] # list of acceptable extensions
bad_file_list, bad_ext_list=check_images(source_dir, good_exts)
if len(bad_file_list) !=0:
    print('improper image files are listed below')
    for i in range (len(bad_file_list)):
        print (bad_file_list[i])
else:
    print(' no improper image files were found')

for i in range(len(bad_file_list)):
    img = Image.open(bad_file_list[i])
    img.save( bad_file_list[i][:-4]+".png", "PNG")
    os.remove(bad_file_list[i]) 

# img = Image.open(train_data_path + "/Fire/fire1.jpg")
# img.save( train_data_path + "/Fire/fire1.png", "PNG")
# os.remove(train_data_path + "/Fire/fire1.jpg") 