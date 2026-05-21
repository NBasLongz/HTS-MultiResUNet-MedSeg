import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model

def polish_heatmap(h, shape, gamma=0.4):
    h = cv2.resize(h, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    h = cv2.GaussianBlur(h, (25, 25), 0)
    return np.power((h - h.min()) / (h.max() - h.min() + 1e-8), gamma)

def get_error_dist(gt, pred):
    diff = cv2.absdiff((gt > 0.5).astype(np.uint8), (pred > 0.5).astype(np.uint8))
    err = cv2.distanceTransform(diff, cv2.DIST_L2, 5)
    err = cv2.GaussianBlur(err, (11, 11), 0)
    return (err - err.min()) / (err.max() - err.min() + 1e-8)

def generate_xai_figure(model, img, h, w, c, save_path="Fig1_XAI.png"):
    img_tensor = np.expand_dims(img, 0).astype(np.float32)
    pred, feat, attn = model.predict(img_tensor, verbose=0)
    
    grad_model = Model(model.inputs, [model.get_layer("cnn_bottleneck_output").output, model.get_layer("final_output").output])
    with tf.GradientTape() as tape:
        conv, p = grad_model(img_tensor); loss = tf.reduce_mean(p)
    grads = tape.gradient(loss, conv); pooled = tf.reduce_mean(grads, axis=(0,1,2))
    
    cnn_h = polish_heatmap(np.maximum(np.squeeze(conv[0] @ pooled[..., None]), 0), (h, w), gamma=0.5)
    trans_h = polish_heatmap(np.max(np.mean(attn[0], axis=1), axis=0).reshape(feat.shape[1], feat.shape[2]), (h, w), gamma=0.5)

    img_u8 = np.uint8(img * 255) if c==3 else cv2.cvtColor(np.uint8(img.squeeze()*255), cv2.COLOR_GRAY2RGB)
    
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(img_u8); ax[0].set_title("Input Image"); ax[0].axis('off')
    ax[1].imshow(cv2.addWeighted(img_u8, 0.4, cv2.applyColorMap(np.uint8(cnn_h*255), cv2.COLORMAP_JET), 0.6, 0)); ax[1].set_title("CNN Local Features"); ax[1].axis('off')
    ax[2].imshow(cv2.addWeighted(img_u8, 0.4, cv2.applyColorMap(np.uint8(trans_h*255), cv2.COLORMAP_JET), 0.6, 0)); ax[2].set_title("Transformer Attention"); ax[2].axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] XAI figure saved to {save_path}")

def generate_error_map(model, img, gt, save_path="Fig3_ErrorMap.png"):
    img_tensor = np.expand_dims(img, 0).astype(np.float32)
    pred_raw, _, _ = model.predict(img_tensor, verbose=0)
    
    err = get_error_dist(gt.squeeze(), pred_raw.squeeze())
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(gt.squeeze(), cmap='gray'); ax[0].set_title("Ground Truth"); ax[0].axis('off')
    ax[1].imshow(img.squeeze(), cmap='gray'); ax[1].set_title("Original Image"); ax[1].axis('off')
    ax[2].imshow(pred_raw.squeeze() > 0.5, cmap='gray'); ax[2].set_title("HTS Output"); ax[2].axis('off')
    im = ax[3].imshow(err, cmap='hot'); ax[3].set_title("Edge Error Map"); ax[3].axis('off')
    plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Error map saved to {save_path}")