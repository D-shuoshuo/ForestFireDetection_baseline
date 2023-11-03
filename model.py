import tensorflow as tf

def FireNet_v1(img_height=64, img_width=64, num_classes=2):
    # NHWC
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu',),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(32, 3, activation='relu',),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, 3, activation='relu',),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    return model

def FireNet_v2(img_height=64, img_width=64, num_classes=2):
    # NHWC
    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, activation='relu',),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(32, 3, activation='relu',),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Conv2D(64, 3, activation='relu',),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model

# def AlexNet_v1(img_height=224, img_width=224, num_classes=1000):
#     # tensorflow中的tensor通道排序是NHWC
#     input_image = tf.keras.layers.Input(shape=(img_height, img_width, 3), dtype="float32")  # output(None, 224, 224, 3)
#     x = tf.keras.layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)                      # output(None, 227, 227, 3)
#     x = tf.keras.layers.Conv2D(48, kernel_size=11, strides=4, activation="relu")(x)       # output(None, 55, 55, 48)
#     x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 27, 27, 48)
#     x = tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", activation="relu")(x)  # output(None, 27, 27, 128)
#     x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 13, 13, 128)
#     x = tf.keras.layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
#     x = tf.keras.layers.Conv2D(192, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 192)
#     x = tf.keras.layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)  # output(None, 13, 13, 128)
#     x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)                              # output(None, 6, 6, 128)

#     x = tf.keras.layers.Flatten()(x)                         # output(None, 6*6*128)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
#     x = tf.keras.layers.Dropout(0.2)(x)
#     x = tf.keras.layers.Dense(2048, activation="relu")(x)    # output(None, 2048)
#     x = tf.keras.layers.Dense(num_classes)(x)                  # output(None, 5)
#     predict = tf.keras.layers.Softmax()(x)

#     model = tf.keras.models.Model(inputs=input_image, outputs=predict)

#     return model


# class AlexNet_v2(Model):
#     def __init__(self, num_classes=1000):
#         super(AlexNet_v2, self).__init__()
#         self.features = Sequential([
#             layers.ZeroPadding2D(((1, 2), (1, 2))),                                 # output(None, 227, 227, 3)
#             layers.Conv2D(48, kernel_size=11, strides=4, activation="relu"),        # output(None, 55, 55, 48)
#             layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 27, 27, 48)
#             layers.Conv2D(128, kernel_size=5, padding="same", activation="relu"),   # output(None, 27, 27, 128)
#             layers.MaxPool2D(pool_size=3, strides=2),                               # output(None, 13, 13, 128)
#             layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
#             layers.Conv2D(192, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 192)
#             layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"),   # output(None, 13, 13, 128)
#             layers.MaxPool2D(pool_size=3, strides=2)])                              # output(None, 6, 6, 128)

#         self.flatten = layers.Flatten()
#         self.classifier = Sequential([
#             layers.Dropout(0.2),
#             layers.Dense(1024, activation="relu"),                                  # output(None, 2048)
#             layers.Dropout(0.2),
#             layers.Dense(128, activation="relu"),                                   # output(None, 2048)
#             layers.Dense(num_classes),                                                # output(None, 5)
#             layers.Softmax()
#         ])

#     def call(self, inputs, **kwargs):
#         x = self.features(inputs)
#         x = self.flatten(x)
#         x = self.classifier(x)
#         return x
