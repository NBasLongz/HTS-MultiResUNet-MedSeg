import tensorflow as tf
import tensorflow.keras.backend as K

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    tp = K.sum(y_true_f * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))
    fp = K.sum((1 - y_true_f) * y_pred_f)
    
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return K.pow((1 - tversky), gamma)

def sobel_edges(y):
    # Khai báo bộ lọc Sobel (3x3)
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)
    sobel_y = tf.reshape(sobel_y, [3, 3, 1, 1])
    
    edge_x = tf.nn.conv2d(y, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    edge_y = tf.nn.conv2d(y, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    edge = tf.sqrt(tf.square(edge_x) + tf.square(edge_y) + K.epsilon())
    return edge

def ea_ftl_loss(y_true, y_pred, lambda_weight=0.5):
    # 1. Region-level loss (FTL)
    ftl = focal_tversky_loss(y_true, y_pred)
    
    # 2. Boundary-level loss (MAE of Sobel edges)
    edge_true = sobel_edges(y_true)
    edge_pred = sobel_edges(y_pred)
    edge_loss = K.mean(K.abs(edge_true - edge_pred))
    
    return ftl + lambda_weight * edge_loss