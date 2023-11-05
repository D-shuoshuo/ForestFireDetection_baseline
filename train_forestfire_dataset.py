import matplotlib.pyplot as plt
from model import ConvNet, FireNet_v1, FireNet_v2, AlexNet_v1
import tensorflow as tf
import os

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 配置信息
# 数据信息
dataset = 'ForestFire_dataset'
img_height = 96
img_width = 96
classes = 2
# 训练信息
batch_size = 32
epochs = 100
binary_class = False
# 文件保存地址信息
saveh5 = 'FireNet_v1_on_ForestFire.h5' # 模型修改
output_figure_dir = 'ForestFire_dataset_output_figure'
loss_fig_name = 'train_FireNet_v1_loss.jpg' # 模型修改
accuracy_fig_name = 'train_FireNet_v1_accuracy.jpg' # 模型修改
# 模型信息
selectedmodel = FireNet_v1 # 模型修改
    

def main():
    # 构造数据集路径
    data_root = os.getcwd()
    # data_root = os.path.abspath(os.path.join(os.getcwd(), '..'))  
    image_path = os.path.join(data_root, 'Dataset', dataset) 
    train_data_path = os.path.join(image_path, 'train') 

    # 创建save_weights文件夹
    if not os.path.exists("save_weights"):  
        os.makedirs("save_weights")

    # 构造用于training和validation的Dataset对象
    train_ds = tf.keras.utils.image_dataset_from_directory(train_data_path,
                                                           label_mode="int",
                                                           validation_split=0.1, 
                                                           subset="training", 
                                                           seed=124,
                                                           image_size=(img_height, img_width),
                                                           batch_size=batch_size)
    
    val_ds = tf.keras.utils.image_dataset_from_directory(train_data_path, 
                                                         label_mode="int",
                                                         validation_split=0.1, 
                                                         subset="validation", 
                                                         seed=124,
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
    
    # plt.figure(figsize=(10, 10))
    # for images, labels in train_ds.take(1):
    #     for i in range(9):
    #         ax = plt.subplot(3, 3, i + 1)
    #         plt.imshow(images[i].numpy().astype("uint8"))
    #         plt.title(class_names[labels[i]])
    # plt.savefig("sample_fig.jpg")

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
                                                     save_best_only=True,
                                                     save_weights_only=False,
                                                     monitor="val_loss")]
    
    history = model.fit(x=train_ds,
                        epochs=epochs,
                        verbose=2,
                        validation_data=val_ds, 
                        callbacks=callbacks)

    # plot loss and accuracy image
    history_dict = history.history
    train_loss = history_dict["loss"]
    train_accuracy = history_dict["accuracy"]
    val_loss = history_dict["val_loss"]
    val_accuracy = history_dict["val_accuracy"]

    # loss figure
    plt.figure()
    plt.plot(range(epochs), train_loss, label='train_loss')
    plt.plot(range(epochs), val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(output_figure_dir + "/" + loss_fig_name)

    # accuracy figure
    plt.figure()
    plt.plot(range(epochs), train_accuracy, label='train_accuracy')
    plt.plot(range(epochs), val_accuracy, label='val_accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.savefig(output_figure_dir + "/" + accuracy_fig_name)








    # # using keras low level api for training
    # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    #
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    #
    # test_loss = tf.keras.metrics.Mean(name='test_loss')
    # test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    #
    #
    # @tf.function
    # def train_step(images, labels):
    #     with tf.GradientTape() as tape:
    #         predictions = model(images, training=True)
    #         loss = loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, model.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #
    #     train_loss(loss)
    #     train_accuracy(labels, predictions)
    #
    #
    # @tf.function
    # def test_step(images, labels):
    #     predictions = model(images, training=False)
    #     t_loss = loss_object(labels, predictions)
    #
    #     test_loss(t_loss)
    #     test_accuracy(labels, predictions)
    #
    #
    # best_test_loss = float('inf')
    # for epoch in range(1, epochs+1):
    #     train_loss.reset_states()        # clear history info
    #     train_accuracy.reset_states()    # clear history info
    #     test_loss.reset_states()         # clear history info
    #     test_accuracy.reset_states()     # clear history info
    #     for step in range(total_train // batch_size):
    #         images, labels = next(train_data_gen)
    #         train_step(images, labels)
    #
    #     for step in range(total_val // batch_size):
    #         test_images, test_labels = next(val_data_gen)
    #         test_step(test_images, test_labels)
    #
    #     template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    #     print(template.format(epoch,
    #                           train_loss.result(),
    #                           train_accuracy.result() * 100,
    #                           test_loss.result(),
    #                           test_accuracy.result() * 100))
    #     if test_loss.result() < best_test_loss:
    #        model.save_weights("./save_weights/myAlex.ckpt", save_format='tf')


if __name__ == '__main__':
    main()
