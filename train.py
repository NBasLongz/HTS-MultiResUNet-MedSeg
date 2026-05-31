import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import KFold
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from models.hts_multiresunet import build_hts_multiresunet
from models.baseline_multires import build_baseline_multiresunet
from utils.losses import ea_ftl_loss
from utils.metrics import dice_coef, iou_coef
from utils.dataset import load_dataset

def get_args():
    parser = argparse.ArgumentParser(description="HTS-MultiResUNet Training Pipeline")
    parser.add_argument('--dataset', type=str, default='isbi2009', 
                        choices=['isbi2012', 'cvc_clinicdb', 'isic2018', 'isbi2009'],
                        help="Dataset name for training")
    parser.add_argument('--image_dir', type=str, required=True,
                        help="Path to images directory")
    parser.add_argument('--mask_dir', type=str, required=True,
                        help="Path to masks directory")
    parser.add_argument('--model_name', type=str, default='hts_multiresunet',
                        choices=['hts_multiresunet', 'baseline_multires'],
                        help="Model architecture selector")
    parser.add_argument('--epochs', type=int, default=150, help="Max number of training epochs")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--n_folds', type=int, default=5, help="Number of folds for Cross-Validation")
    parser.add_argument('--early_stopping', type=int, default=30, help="Patience for early stopping")
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory to save weights and logs")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    args = get_args()
    set_seed(args.seed)
    
    print(f"\n--- START TRAINING: {args.model_name.upper()} ON {args.dataset.upper()} ---")
    
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
    X, Y = load_dataset(
        dataset_name=args.dataset,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        height=H,
        width=W,
        n_channels=C,
        data_color=data_color
    )
    
    # 3. K-Fold Validation
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {
        "fold_max_jaccard": [],
        "fold_max_dice": []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n================ FOLD {fold}/{args.n_folds} ================")
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        K.clear_session()
        
        if args.model_name == 'hts_multiresunet':
            model = build_hts_multiresunet(H, W, C, return_attention=False)
        else:
            model = build_baseline_multiresunet(H, W, C)
            
        model.compile(
            optimizer=Adam(learning_rate=args.lr),
            loss=ea_ftl_loss,
            metrics=[dice_coef, iou_coef, 'accuracy']
        )
        
        ckpt_path = os.path.join(args.save_dir, f"{args.model_name}_{args.dataset}_fold{fold}.weights.h5")
        checkpoint = ModelCheckpoint(
            ckpt_path,
            monitor="val_iou_coef",
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_iou_coef',
            mode='max',
            patience=args.early_stopping,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train fold
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=[checkpoint, reduce_lr, early_stopping],
            verbose=1
        )
        
        model.load_weights(ckpt_path)
        
        # Re-evaluate
        val_loss, val_dice, val_iou, val_acc = model.evaluate(X_val, Y_val, verbose=0)
        
        results["fold_max_jaccard"].append(val_iou)
        results["fold_max_dice"].append(val_dice)
        
        print(f"[RESULT] Fold {fold} - Best Jaccard (IoU): {val_iou*100:.2f}% | Best Dice: {val_dice*100:.2f}%")
        
    print("\n================ K-FOLD SUMMARY ================")
    mean_jaccard = np.mean(results["fold_max_jaccard"]) * 100
    mean_dice = np.mean(results["fold_max_dice"]) * 100
    print(f"Mean Jaccard (IoU) over {args.n_folds}-Fold: {mean_jaccard:.2f}%")
    print(f"Mean Dice Coefficient over {args.n_folds}-Fold: {mean_dice:.2f}%")
    print("==================================================")

if __name__ == '__main__':
    main()