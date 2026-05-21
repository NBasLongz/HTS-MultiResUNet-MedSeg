import argparse
import numpy as np
from models.hts_multiresunet import build_hts_multiresunet
from utils.visualization import generate_xai_figure, generate_error_map

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--visualize', type=bool, default=True)
    return parser.parse_args()

def main():
    args = get_args()
    print(f"--- Evaluating on {args.dataset.upper()} ---")
    
    # ---------------------------------------------------------
    # TODO: Cắm logic load data test của ông vào đây
    # X_test, y_test = load_test_data(args.dataset)
    # Chạy tạm với ảnh dummy để demo
    H, W, C = 256, 256, 3
    X_test = np.random.rand(1, H, W, C)
    y_test = np.random.randint(0, 2, (1, H, W, 1))
    # ---------------------------------------------------------
    
    # Load mô hình và trả về attention map (cho visualization)
    model = build_hts_multiresunet(H, W, C, return_attention=True)
    model.load_weights(args.weights_path)
    print(f"[SUCCESS] Loaded weights from {args.weights_path}")
    
    # Đoạn đánh giá số liệu thực tế ông thêm `model.evaluate()` nếu dùng model bình thường
    
    if args.visualize:
        print("[INFO] Generating visualizations...")
        sample_img = X_test[0]
        sample_gt = y_test[0]
        
        generate_xai_figure(model, sample_img, H, W, C, save_path=f"Fig1_{args.dataset}.png")
        generate_error_map(model, sample_img, sample_gt, save_path=f"Fig3_{args.dataset}.png")

if __name__ == '__main__':
    main()