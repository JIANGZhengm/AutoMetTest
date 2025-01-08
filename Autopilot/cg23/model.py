import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D


# 自定义最后一层的损失函数
def custom_activation(x):
    return tf.multiply(tf.atan(x), 2)


def build_cnn(image_size=None, weights_path=None):
    image_size = image_size or (128, 128)
    input_shape = image_size + (3,)

    img_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    y = Flatten()(x)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1)(y)
    y = Lambda(custom_activation)(y)  # 最后一层的模型输出

    model = Model(inputs=img_input, outputs=y)

    if weights_path:
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(weights_path).expect_partial()

    return model


# 构建冻结部分权重的模型
def build_finetune_model(pretrained_model, trainable_layers=5):
    """
    :param pretrained_model: 已加载权重的模型
    :param trainable_layers: 微调的可训练层数（从模型最后几层开始）
    """
    for layer in pretrained_model.layers[:-trainable_layers]:
        layer.trainable = False  # 冻结大部分权重

    # 在现有模型的基础上添加微调层
    x = pretrained_model.layers[-3].output
    x = Dense(512, activation='relu')(x)  # 添加一个全连接层
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)  # 添加一个全连接层
    x = Dropout(0.25)(x)
    y = Dense(1)(x)
    y = Lambda(custom_activation)(y)  # 保留自定义输出层

    fine_tune_model = Model(inputs=pretrained_model.input, outputs=y)
    return fine_tune_model


def build_InceptionV3(image_size=None, weights_path=None):
    image_size = image_size or (299, 299)
    input_shape = image_size + (3,)
    bottleneck_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in bottleneck_model.layers:
        layer.trainable = False

    x = bottleneck_model.input
    y = bottleneck_model.output
    y = GlobalAveragePooling2D()(y)
    y = Dense(1024, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(1)(y)
    y = Lambda(custom_activation)(y)

    model = Model(inputs=x, outputs=y)

    if weights_path:
        model.load_weights(weights_path)

    return model
