from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu'):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)
    if activation: x = Activation(activation)(x)
    return x

def build_baseline_multiresunet(height, width, n_channels):
    def MultiResBlock(U, inp): 
        W = 1.67 * U
        short = conv2d_bn(inp, int(W*0.167)+int(W*0.333)+int(W*0.5), 1, 1, activation=None)
        c3 = conv2d_bn(inp, int(W*0.167), 3, 3)
        c5 = conv2d_bn(c3, int(W*0.333), 3, 3)
        c7 = conv2d_bn(c5, int(W*0.5), 3, 3)
        return Activation('relu')(add([short, concatenate([c3, c5, c7], 3)]))

    def ResPath(f, length, inp): 
        out = conv2d_bn(inp, f, 3, 3)
        out = add([conv2d_bn(inp, f, 1, 1, activation=None), out])
        out = Activation('relu')(BatchNormalization(axis=3)(out))
        for _ in range(length - 1):
            short = conv2d_bn(out, f, 1, 1, activation=None)
            out = conv2d_bn(out, f, 3, 3)
            out = Activation('relu')(add([short, out]))
            out = BatchNormalization(axis=3)(out)
        return out

    inputs = Input((height, width, n_channels))
    m1 = MultiResBlock(32, inputs); p1 = MaxPooling2D((2,2))(m1); r1 = ResPath(32, 4, m1)
    m2 = MultiResBlock(64, p1); p2 = MaxPooling2D((2,2))(m2); r2 = ResPath(64, 3, m2)
    m3 = MultiResBlock(128, p2); p3 = MaxPooling2D((2,2))(m3); r3 = ResPath(128, 2, m3)
    m4 = MultiResBlock(256, p3); p4 = MaxPooling2D((2,2))(m4); r4 = ResPath(256, 1, m4)
    m5 = MultiResBlock(512, p4)
    
    u6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(m5), r4], 3); m6 = MultiResBlock(256, u6)
    u7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(m6), r3], 3); m7 = MultiResBlock(128, u7)
    u8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(m7), r2], 3); m8 = MultiResBlock(64, u8)
    u9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(m8), r1], 3); m9 = MultiResBlock(32, u9)
    
    return Model(inputs, Conv2D(1, (1,1), activation='sigmoid')(m9))