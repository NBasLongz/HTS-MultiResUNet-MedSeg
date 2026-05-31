import os
import re
import glob
import cv2
import numpy as np
from tqdm import tqdm

try:
    import tifffile
except ImportError:
    tifffile = None

def extract_main_number(path):
    """
    Trích xuất số ID cuối cùng từ tên file để map giữa ảnh gốc và masks.
    Ví dụ: 'ISIC_0000000_segmentation.png' -> '0000000'
    """
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r'\d+', name)
    if not nums:
        return None
    return nums[-1]

def load_dataset(dataset_name, image_dir, mask_dir, height, width, n_channels=3, data_color="RGB"):
    """
    Hàm load dataset dùng chung cho 4 bộ dữ liệu:
    isbi2012, isbi2009, cvc_clinicdb, isic2018
    """
    dataset_name = dataset_name.lower().strip()
    assert data_color in ["RGB", "BGR", "Grayscale"], "Invalid color space"
    
    print(f"[INFO] Loading dataset: {dataset_name.upper()}")
    print(f"[INFO] Image directory: {image_dir}")
    print(f"[INFO] Mask directory: {mask_dir}")

    # Trường hợp đặc biệt: ISBI-2012 đọc file .tif 3D (single file volume)
    if dataset_name == "isbi2012" and os.path.isfile(image_dir):
        if tifffile is None:
            raise ImportError(
                "Please install 'tifffile' to read ISBI-2012 3D TIFF files: pip install tifffile"
            )
        imgs = tifffile.imread(image_dir)
        masks = tifffile.imread(mask_dir)
        image_paths = list(range(len(imgs)))
        mask_paths = list(range(len(masks)))
        mask_dict = {i: i for i in mask_paths}
        is_tif_file = True
    else:
        # Đọc dữ liệu thư mục thông thường
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        
        if dataset_name in ["isbi2009", "isic2018", "isbi2012"]:
            mask_dict = {extract_main_number(p): p for p in mask_paths}
        elif dataset_name == "cvc_clinicdb":
            mask_dict = {os.path.basename(p): p for p in mask_paths}
        else:
            # Fallback map mặc định theo tên file nếu không xác định
            mask_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in mask_paths}
        is_tif_file = False

    X, Y = [], []

    for img_path in tqdm(image_paths, desc=f"Loading {dataset_name.upper()}"):
        # Lấy khóa map
        if is_tif_file:
            key = img_path
        else:
            if dataset_name in ["isbi2009", "isic2018", "isbi2012"]:
                key = extract_main_number(img_path)
            elif dataset_name == "cvc_clinicdb":
                key = os.path.basename(img_path)
            else:
                key = os.path.splitext(os.path.basename(img_path))[0]

        if key not in mask_dict:
            continue

        # 1. Đọc và tiền xử lý ảnh gốc
        if is_tif_file:
            img = imgs[img_path]
        else:
            if data_color == "Grayscale":
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            if img is None:
                continue

            if data_color == "RGB":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif data_color == "Grayscale" and img.ndim == 2:
                img = np.expand_dims(img, -1)

        # Resize ảnh gốc
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        if img.ndim == 2:
            img = np.stack([img] * n_channels, axis=-1)
        elif img.ndim == 3 and img.shape[-1] != n_channels:
            if n_channels == 1:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = np.expand_dims(img, -1)
            elif n_channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = img.astype(np.float32) / 255.0

        # 2. Đọc và tiền xử lý mask
        if is_tif_file:
            mask = masks[mask_dict[key]]
        else:
            mask = cv2.imread(mask_dict[key], cv2.IMREAD_GRAYSCALE)
            
        if mask is None:
            continue

        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = np.round(mask.astype(np.float32) / 255.0, 0)
        mask = np.expand_dims(mask, -1)

        X.append(img)
        Y.append(mask)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"[SUCCESS] Loaded dataset successfully. X shape: {X.shape} | Y shape: {Y.shape}")
    return X, Y
