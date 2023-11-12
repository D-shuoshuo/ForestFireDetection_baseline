import os

import tensorflow as tf
import matplotlib.pyplot as plt

from model import ConvNet, FireNet_v1, FireNet_v2, AlexNet_v1

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 配置信息
# 数据信息
dataset = 'ForestFire_dataset'
img_height = 224 # 输入图片大小
img_width = 224
classes = 2
# 训练信息
batch_size = 32
epochs = 100
binary_class = False
# 文件保存地址信息
saveh5 = 'ConvNet_on_ForestFire_224.h5' # 图片大小与模型修改
output_figure_dir = 'ForestFire_dataset_output_figure'
loss_fig_name = 'train_ConvNet_loss_224.jpg' # 图片大小与模型修改
accuracy_fig_name = 'train_ConvNet_accuracy_224.jpg' # 图片大小与模型修改
# 模型信息
selectedmodel = ConvNet # 模型修改


def main():
    # 构造数据集路径
    data_root = os.getcwd()
    # data_root = os.path.abspath(os.path.join(os.getcwd(), '..')) 
    image_path = os.path.join(data_root, "Dataset", dataset) 
    train_data_path = os.path.join(image_path, "train") 

    # 创建save_weights文件夹
    if not os.path.exists("save_weights"):  
        os.makedirs("save_weights")

    # 构造用于training和validation的Dataset对象
    train_ds = tf.keras.utils.image_dataset_from_directory(train_data_path,
                                                           label_mode='int',
                                                           validation_split=0.2, 
                                                           subset="training", 
                                                           seed=111,
                                                           image_size=(img_height, img_width),
                                                           batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(train_data_path, 
                                                         label_mode='int',
                                                         validation_split=0.2, 
                                                         subset="validation", 
                                                         seed=111,
                                                         image_size=(img_height, img_width),
                                                         batch_size=batch_size)
    # 查看train_ds的信息
    class_names = train_ds.class_names
    print("Classs Names:", class_names)
    
    num_batches = len(train_ds)  
    print("Number of batches in the training dataset:", num_batches)

    for image_batch, labels_batch in train_ds:
            print(image_batch.shape)
            print(labels_batch.shape)
            break
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
    plt.savefig("sample_fig.jpg")


    # # 配置数据集以提高性能
    # AUTOTUNE = tf.data.AUTOTUNE

    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    if binary_class:
         num_classes = 1
    else:
         num_classes = classes

    model = selectedmodel(img_height=img_height, 
                          img_width=img_width, 
                          num_classes=num_classes, 
                          steps_per_epoch=num_batches)

    # model.build((batch_size, 224, 224, 3))  # when using subclass model
    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./save_weights/"+saveh5,
                                                     save_best_only=False,
                                                     save_weights_only=False,
                                                     monitor='val_loss')]
    
    history = model.fit(x=train_ds,
                        epochs=epochs,
                        verbose=2,
                        validation_data=val_ds, 
                        callbacks=callbacks)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict['loss']
    train_accuracy = history_dict['accuracy']
    val_loss = history_dict['val_loss']
    val_accuracy = history_dict['val_accuracy']

    # loss figure
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(output_figure_dir + '/' + loss_fig_name)

    # accuracy figure
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(output_figure_dir + '/' + accuracy_fig_name)


if __name__ == '__main__':
    main()
