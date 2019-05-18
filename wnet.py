from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, multiply, BatchNormalization, ReLU, Activation
from keras.models import Model
from keras.initializers import RandomNormal


def wnet(input_shape=(None, None, 3), BN=False):
    # Difference with original paper: padding 'valid vs same'
    conv_kernel_initializer = RandomNormal(stddev=0.01)

    input_flow = Input(input_shape)
    # Encoder
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(input_flow)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_1 = BatchNormalization()(x_1) if BN else x_1
    x_1 = Activation('relu')(x_1)

    x = MaxPooling2D((2, 2))(x_1)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_2 = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_2 = BatchNormalization()(x_2) if BN else x_2
    x_2 = Activation('relu')(x_2)

    x = MaxPooling2D((2, 2))(x_2)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_3 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_3 = BatchNormalization()(x_3) if BN else x_3
    x_3 = Activation('relu')(x_3)

    x = MaxPooling2D((2, 2))(x_3)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x_4 = Conv2D(512, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x_4 = BatchNormalization()(x_4) if BN else x_4
    x_4 = Activation('relu')(x_4)

    # Decoder 1
    x = UpSampling2D((2, 2))(x_4)
    x = concatenate([x_3, x])
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x_2, x])
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = concatenate([x_1, x])
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x)
    x = BatchNormalization()(x) if BN else x
    x = Activation('relu')(x)

    # Decoder 2
    x_rb = UpSampling2D((2, 2))(x_4)
    x_rb = concatenate([x_3, x_rb])
    x_rb = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)

    x_rb = UpSampling2D((2, 2))(x_rb)
    x_rb = concatenate([x_2, x_rb])
    x_rb = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)

    x_rb = UpSampling2D((2, 2))(x_rb)
    x_rb = concatenate([x_1, x_rb])
    x_rb = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer)(x_rb)
    x_rb = BatchNormalization()(x_rb) if BN else x_rb
    x_rb = Activation('relu')(x_rb)
    x_rb = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='sigmoid')(x_rb)    # Sigmoid activation

    # Multiplication
    x = multiply([x, x_rb])
    x = Conv2D(1, (1, 1), strides=(1, 1), padding='same', kernel_initializer=conv_kernel_initializer, activation='relu')(x)

    model = Model(inputs=input_flow, outputs=x)

    front_end = VGG16(weights='imagenet', include_top=False)
    weights_front_end = []
    for layer in front_end.layers:
        if 'conv' in layer.name:
            weights_front_end.append(layer.get_weights())
    counter_conv = 0
    for i in range(len(model.layers)):
        if counter_conv >= 13:
            break
        if 'conv' in model.layers[i].name:
            model.layers[i].set_weights(weights_front_end[counter_conv])
            counter_conv += 1

    return model
