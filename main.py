import os

import tensorflow as tf

from train.train_firenet_dataset import train_on_FireNet_dataset
from test.test_firenet_dataset import test_on_FireNet_dataset
from train.train_forestfire_dataset import train_on_ForestFire_dataset
from test.test_forestfire_dataset import test_on_ForestFire_dataset
from train.train_FLAME import train_on_FLAME
from test.test_FLAME import test_on_FLAME
from model.model import ConvNet, FireNet_v1, Tiny_Xception

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))


# 配置信息
# 数据信息
img_height = 224 # 输入图片大小
img_width = 224
classes = 2
validation_split = 0.2
# 训练信息
batch_size = 32
epochs = 100
binary_class = False
# 模型信息
selected_dataset = "FLAME" # FireNet ForestFire FLAME
target_model = "Tiny_Xception" # ConvNet, FireNet_v1, Tiny_Xception

def main():
    # create a "save_weights" folder
    if not os.path.exists("save_weights"):
        os.makedirs("save_weights")

    # select model
    models = [ConvNet, FireNet_v1, Tiny_Xception]
    for model in models:
        if model.__name__ == target_model:
            selectedmodel = model
            break

    if selected_dataset == "FireNet":
        train_on_FireNet_dataset(selectedmodel=selectedmodel,
                                 batch_size=batch_size,  
                                 epochs=epochs,
                                 validation_split=validation_split, 
                                 img_height=img_height, 
                                 img_width=img_width,
                                 binary_class=binary_class,
                                 classes = classes)  
        test_on_FireNet_dataset(selectedmodel=selectedmodel,
                                batch_size=batch_size,  
                                img_height=img_height, 
                                img_width=img_width,)
        
    elif selected_dataset == "ForestFire":
        train_on_ForestFire_dataset(selectedmodel=selectedmodel,
                                    batch_size=batch_size,  
                                    epochs=epochs,
                                    validation_split=validation_split, 
                                    img_height=img_height, 
                                    img_width=img_width,
                                    binary_class=binary_class,
                                    classes = classes)  
        test_on_ForestFire_dataset(selectedmodel=selectedmodel,
                                   batch_size=batch_size,  
                                   img_height=img_height, 
                                   img_width=img_width)
        
    elif selected_dataset == "FLAME":
        train_on_FLAME(selectedmodel=selectedmodel,
                       batch_size=batch_size,  
                       epochs=epochs,
                       validation_split=validation_split, 
                       img_height=img_height, 
                       img_width=img_width,
                       binary_class=binary_class,
                       classes = classes)  
        test_on_FLAME(selectedmodel=selectedmodel,
                      batch_size=batch_size,  
                      img_height=img_height, 
                      img_width=img_width)
    else:
        assert "There is no dataset matching {}.".format(selected_dataset)


if __name__ == "__main__":
    main()
