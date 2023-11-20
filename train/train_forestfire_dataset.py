import os

import tensorflow as tf
import matplotlib.pyplot as plt


# 数据集信息
classes = 2
# 文件保存地址信息
saveh5 = '_on_ForestFire_dataset.h5' # 图片大小与模型修改
output_figure_dir = 'ForestFire_dataset_output_figure'


def train_on_ForestFire_dataset(selectedmodel, 
                             batch_size=32,  
                             epochs=100,
                             validation_split=0.2, 
                             img_height=224,
                             img_width=224,
                             binary_class=False):
    # 构造数据集路径
    data_root = os.getcwd()
    train_data_path = os.path.join(data_root, "Dataset/ForestFire_dataset/train")

    # 构造文件存储地址
    dataset_saveh5 = str(img_height) + "_" + selectedmodel.__name__ + saveh5
    dataset_loss_fig_name = str(img_height) + "_" + selectedmodel.__name__ + '_loss.jpg'
    dataset_accuracy_fig_name = str(img_height) + "_" + selectedmodel.__name__ + '_accuracy.jpg'

    # 构造用于training和validation的Dataset对象
    train_ds = tf.keras.utils.image_dataset_from_directory(train_data_path,
                                                           label_mode='int',
                                                           validation_split=validation_split, 
                                                           subset="training", 
                                                           seed=111,
                                                           image_size=(img_height, img_width),
                                                           batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(train_data_path, 
                                                         label_mode='int',
                                                         validation_split=validation_split, 
                                                         subset="validation", 
                                                         seed=111,
                                                         image_size=(img_height, img_width),
                                                         batch_size=batch_size)
    # 查看train_ds的信息
    class_names = train_ds.class_names
    print("="*40)
    print("Classs Names:", class_names)
    print("="*40)
    
    num_batches = len(train_ds)
    print("="*40) 
    print("Number of batches in the training dataset:", num_batches)
    print("="*40)

    for image_batch, labels_batch in train_ds:
            print("="*40)
            print(image_batch.shape)
            print(labels_batch.shape)
            print("="*40)
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

    model.summary()

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath="./save_weights/"+dataset_saveh5,
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
    plt.savefig("figure/" + output_figure_dir + '/' + dataset_loss_fig_name)

    # accuracy figure
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig("figure/" + output_figure_dir + '/' + dataset_accuracy_fig_name)

