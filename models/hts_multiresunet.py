import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose, 
                                     concatenate, BatchNormalization, Activation, add,
                                     MultiHeadAttention, LayerNormalization, Dense, Reshape,
                                     GlobalAveragePooling2D, Multiply)

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu'):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x) # IMPORTANT: scale=False for strict weight loading
    if activation: x = Activation(activation)(x)
    return x

def squeeze_excite_block(inputs, ratio=8):
    c = K.int_shape(inputs)[3]
    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, c))(x)
    x = Dense(max(c // ratio, 1), activation='relu', use_bias=False)(x)
    x = Dense(c, activation='sigmoid', use_bias=False)(x)
    return Multiply()([inputs, x])

def build_hts_multiresunet(height, width, n_channels, return_attention=False):
    def MultiResBlock_hts(U, inp, name=None):
        W = 1.67 * U
        f1, f2, f3 = int(W*0.167), int(W*0.333), int(W*0.5)
        shortcut = conv2d_bn(inp, f1 + f2 + f3, 1, 1, activation=None)
        c3 = conv2d_bn(inp, f1, 3, 3); c5 = conv2d_bn(c3, f2, 3, 3); c7 = conv2d_bn(c5, f3, 3, 3)
        out = BatchNormalization(axis=3)(concatenate([c3, c5, c7], 3))
        out = Activation('relu')(add([shortcut, out]))
        out = squeeze_excite_block(out)
        if name: out = Activation('linear', name=name)(out)
        return out

    def ResPath_hts(filters, length, inp):
        shortcut = conv2d_bn(inp, filters, 1, 1, activation=None)
        out = conv2d_bn(inp, filters, 3, 3)
        out = add([shortcut, out])
        out = BatchNormalization(axis=3)(out)
        out = Activation('relu')(out)
        for _ in range(length - 1):
            shortcut = conv2d_bn(out, filters, 1, 1, activation=None)
            conv = conv2d_bn(out, filters, 3, 3)
            out = add([shortcut, conv])
            out = BatchNormalization(axis=3)(out)
            out = Activation('relu')(out)
        return squeeze_excite_block(out)

    inputs = Input((height, width, n_channels))
    m1 = MultiResBlock_hts(32, inputs); p1 = MaxPooling2D((2,2))(m1); r1 = ResPath_hts(32, 4, m1)
    m2 = MultiResBlock_hts(64, p1); p2 = MaxPooling2D((2,2))(m2); r2 = ResPath_hts(64, 3, m2)
    m3 = MultiResBlock_hts(128, p2); p3 = MaxPooling2D((2,2))(m3); r3 = ResPath_hts(128, 2, m3)
    m4 = MultiResBlock_hts(256, p3); p4 = MaxPooling2D((2,2))(m4); r4 = ResPath_hts(256, 1, m4)
    m5 = MultiResBlock_hts(512, p4, name='cnn_bottleneck_output') 
    
    hf, wf, cf = K.int_shape(m5)[1], K.int_shape(m5)[2], K.int_shape(m5)[3]
    x = Reshape((hf * wf, cf))(m5)
    attn_out, attn_scores = MultiHeadAttention(num_heads=8, key_dim=64)(x, x, return_attention_scores=True)
    x = LayerNormalization(epsilon=1e-6)(add([x, attn_out]))
    ffn = Dense(cf)(Dense(1024, activation='relu')(x))
    x = LayerNormalization(epsilon=1e-6)(add([x, ffn]))
    trans_bottleneck = Reshape((hf, wf, cf))(x)

    u6 = concatenate([Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(trans_bottleneck), r4], 3); m6 = MultiResBlock_hts(256, u6)
    u7 = concatenate([Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(m6), r3], 3); m7 = MultiResBlock_hts(128, u7)
    u8 = concatenate([Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(m7), r2], 3); m8 = MultiResBlock_hts(64, u8)
    u9 = concatenate([Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(m8), r1], 3); m9 = MultiResBlock_hts(32, u9)
    
    output = Conv2D(1, (1,1), activation='sigmoid', name='final_output')(m9)
    
    if return_attention:
        return Model(inputs, [output, m5, attn_scores])
    return Model(inputs, output)