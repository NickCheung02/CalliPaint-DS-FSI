import numpy as np
import os
import cv2
import argparse
import math

def create_heatmap_grid(npz_path, save_dir, cols=4, padding=10, save_individual=False):
    print(f"Processing {os.path.basename(npz_path)}...")
    
    try:
        data = np.load(npz_path)
        sorted_keys = sorted(data.files)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 1. 准备所有子图
    processed_imgs = []
    base_size = 512 # 基础分辨率
    
    # 定义字体 (OpenCV 内置)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    text_padding = 60 # 顶部给文字留黑边的通过高度

    for name in sorted_keys:
        heatmap = data[name]
        
        # 归一化 (0-255)
        min_v, max_v = heatmap.min(), heatmap.max()
        if max_v > min_v:
            norm = (heatmap - min_v) / (max_v - min_v)
        else:
            norm = np.zeros_like(heatmap)
        norm = (norm * 255).astype(np.uint8)
        
        # Resize
        img = cv2.resize(norm, (base_size, base_size), interpolation=cv2.INTER_NEAREST)
        # 伪彩色
        color_img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        
        # 添加标签栏 (顶部黑色区域)
        # 简化层名: "05_Output_Block_3" -> "Out_Blk_3" 以节省空间
        label = name.split('_', 1)[1] if '_' in name else name
        label = label.replace("Input", "In").replace("Output", "Out").replace("Middle", "Mid").replace("Block", "Blk")
        
        labeled_img = np.zeros((base_size + text_padding, base_size, 3), dtype=np.uint8)
        labeled_img[:text_padding, :] = (0, 0, 0) # 黑底
        labeled_img[text_padding:, :] = color_img # 图内容
        
        # 居中写字
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = (base_size - text_size[0]) // 2
        text_y = (text_padding + text_size[1]) // 2
        cv2.putText(labeled_img, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
        
        processed_imgs.append(labeled_img)

        # (可选) 单独保存
        if save_individual:
            ind_dir = os.path.join(save_dir, "individual")
            os.makedirs(ind_dir, exist_ok=True)
            ind_name = f"{os.path.basename(npz_path).replace('.npz','')}_{name}.jpg"
            cv2.imwrite(os.path.join(ind_dir, ind_name), labeled_img)

    # 2. 网格拼接逻辑
    N = len(processed_imgs)
    cols = min(cols, N)
    rows = math.ceil(N / cols)
    
    # 计算总画布大小 (包含 padding)
    # 单元格大小: (H, W) = (base_size + text_padding, base_size)
    cell_h = base_size + text_padding
    cell_w = base_size
    
    total_w = cols * cell_w + (cols - 1) * padding
    total_h = rows * cell_h + (rows - 1) * padding
    
    # 创建白底画布
    grid_canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255 
    
    for idx, img in enumerate(processed_imgs):
        r = idx // cols
        c = idx % cols
        
        y_start = r * (cell_h + padding)
        x_start = c * (cell_w + padding)
        
        grid_canvas[y_start:y_start+cell_h, x_start:x_start+cell_w] = img

    # 3. 保存
    layout_name = f"grid_{rows}x{cols}"
    filename = os.path.basename(npz_path).replace('_features.npz', '')
    save_path = os.path.join(save_dir, f"{filename}_{layout_name}.jpg")
    
    cv2.imwrite(save_path, grid_canvas)
    print(f"SUCCESS: Saved {layout_name} to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./test-result/feature_data_v1/feature_data_arrays')
    parser.add_argument('--save_dir', type=str, default='./test-result/feature_data_v1/plots_paper')
    # 核心参数: 每行放几张图?
    parser.add_argument('--cols', type=int, default=4, help='每行显示的图片数量 (例如: 8=跨栏长图, 4=2行4列, 2=单栏长图)')
    parser.add_argument('--padding', type=int, default=0, help='图片之间的白色间距(像素)')
    parser.add_argument('--save_individual', action='store_true', help='是否保存单张图片以便在LaTeX中手动排版')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if os.path.exists(args.data_dir):
        files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
        if not files:
            print("No .npz files found.")
        for f in files:
            create_heatmap_grid(os.path.join(args.data_dir, f), args.save_dir, 
                                cols=args.cols, 
                                padding=args.padding,
                                save_individual=args.save_individual)
    else:
        print(f"Data directory not found: {args.data_dir}")