import argparse
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from models.hts_multiresunet import build_hts_multiresunet
from utils.losses import ea_ftl_loss
from utils.metrics import dice_coef, iou_coef

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cvc_clinicdb')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser.parse_args()

def main():
    args = get_args()
    print(f"--- Training HTS-MultiResUNet on {args.dataset.upper()} ---")
    
    # ---------------------------------------------------------
    # TODO: Cắm logic load data của ông vào đây
    # X_train, y_train, X_val, y_val = load_data_function(args.dataset)
    # ---------------------------------------------------------
    
    # Dummy shapes based on common dataset config
    H, W, C = 256, 256, 3 
    
    model = build_hts_multiresunet(H, W, C, return_attention=False)
    
    model.compile(
        optimizer=AdamW(learning_rate=args.lr),
        loss=ea_ftl_loss,
        metrics=[dice_coef, iou_coef, 'accuracy']
    )
    
    callbacks = [
        ModelCheckpoint(f"weights_{args.dataset}.h5", save_best_only=True, monitor='val_iou_coef', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    print("[INFO] Model compiled. Ready for training.")
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks)

if __name__ == '__main__':
    main()