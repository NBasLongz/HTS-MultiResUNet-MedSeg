import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from models.hts_multiresunet import build_hts_multiresunet
from models.baseline_multires import build_baseline_multiresunet
from utils.losses import ea_ftl_loss
from utils.metrics import dice_coef, iou_coef
from utils.dataset import load_dataset
from utils.visualization import generate_xai_figure, generate_error_map

def get_args():
    parser = argparse.ArgumentParser(description="HTS-MultiResUNet Evaluation Pipeline")
    parser.add_argument('--dataset', type=str, default='isbi2009',
                        choices=['isbi2012', 'cvc_clinicdb', 'isic2018', 'isbi2009'],
                        help="Dataset name for evaluation")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Path to test images directory")
    parser.add_argument('--mask_dir', type=str, required=True,
                        help="Path to test masks directory")
    parser.add_argument('--weights_path', type=str, required=True,
                        help="Path to saved model weights file (.weights.h5 or .h5)")
    parser.add_argument('--model_name', type=str, default='hts_multiresunet',
                        choices=['hts_multiresunet', 'baseline_multires'],
                        help="Model architecture selector")
    parser.add_argument('--visualize', type=str, default='True',
                        help="Whether to generate XAI and Error maps (True/False)")
    parser.add_argument('--save_dir', type=str, default='./results',
                        help="Directory to save visualizer output images")
    return parser.parse_args()

def main():
    args = get_args()
    visualize = args.visualize.lower() == 'true'
    
    print(f"\n--- START EVALUATION: {args.model_name.upper()} ON {args.dataset.upper()} ---")
    
    # 1. Image configurations
    if args.dataset == 'isbi2012':
        H, W, C = 256, 256, 1
        data_color = "Grayscale"
    elif args.dataset == 'cvc_clinicdb':
        H, W, C = 192, 256, 3
        data_color = "RGB"
    elif args.dataset == 'isic2018':
        H, W, C = 192, 256, 3
        data_color = "RGB"
    else: # isbi2009
        H, W, C = 256, 256, 3
        data_color = "RGB"

    # 2. Load dataset
    X_test, y_test = load_dataset(
        dataset_name=args.dataset,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        height=H,
        width=W,
        n_channels=C,
        data_color=data_color
    )
    
    # 3. Build model and load weights
    K.clear_session()
    
    if args.model_name == 'hts_multiresunet':
        model = build_hts_multiresunet(H, W, C, return_attention=visualize)
    else:
        model = build_baseline_multiresunet(H, W, C)
        
    model.load_weights(args.weights_path)
    print(f"[SUCCESS] Loaded weights successfully from {args.weights_path}")
    
    # 4. Evaluation metrics
    if visualize and args.model_name == 'hts_multiresunet':
        preds, _, _ = model.predict(X_test, batch_size=4, verbose=1)
    else:
        preds = model.predict(X_test, batch_size=4, verbose=1)
        
    jaccards, dices = [], []
    for i in range(len(y_test)):
        y_true_f = y_test[i].flatten()
        y_pred_f = (preds[i].flatten() > 0.5).astype(np.float32)
        
        intersection = np.sum(y_true_f * y_pred_f)
        union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
        
        jaccard = (intersection + 1e-6) / (union + 1e-6)
        dice = (2. * intersection + 1e-6) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1e-6)
        
        jaccards.append(jaccard)
        dices.append(dice)
        
    mean_jaccard = np.mean(jaccards) * 100
    mean_dice = np.mean(dices) * 100
    
    print("\n================ EVALUATION ON TEST SET ================")
    print(f"Mean Jaccard Coefficient (IoU): {mean_jaccard:.2f}%")
    print(f"Mean Dice Coefficient (F1-score): {mean_dice:.2f}%")
    print("==========================================================")
    
    # 5. Visualizer
    if visualize:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"[INFO] Generating visualizer outputs in {args.save_dir}...")
        
        sample_img = X_test[0]
        sample_gt = y_test[0]
        
        if args.model_name == 'hts_multiresunet':
            xai_path = os.path.join(args.save_dir, f"Fig1_XAI_{args.dataset}.png")
            generate_xai_figure(model, sample_img, H, W, C, save_path=xai_path)
            
        error_path = os.path.join(args.save_dir, f"Fig3_ErrorMap_{args.dataset}.png")
        if args.model_name == 'hts_multiresunet':
            eval_model = build_hts_multiresunet(H, W, C, return_attention=False)
            eval_model.load_weights(args.weights_path)
        else:
            eval_model = model
            
        generate_error_map(eval_model, sample_img, sample_gt, save_path=error_path)
        print("[SUCCESS] Visualization generated successfully.")

if __name__ == '__main__':
    main()