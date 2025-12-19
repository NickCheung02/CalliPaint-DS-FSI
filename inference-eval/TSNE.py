import sys
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- 路径设置 ---
# 确保能导入 cldm 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cldm.model import create_model, load_state_dict
from dataset_util import load

# --- 1. 用户配置参数 (已更新) ---
config_yaml = './models_yaml/anytext2_sd15.yaml'
ckpt_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/checkpoints/lightning_logs/version_1/checkpoints/epoch=24-step=900.ckpt'
json_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/data/data4_RESULTS2_WithFit/RESULTS2_WithFit.json'
output_jpg = 'tsne_analysis_metrics.jpg'

max_items = 100  # 分析样本数 (如果样本少于此数，会自动调整)
use_fp16 = True
CONTEXT_PROMPT_KEYS = ['full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']

# 颜色定义 (RGB)
COLORS = {
    'full_prompt': (255, 0, 0),    # 红
    'element_prompt': (0, 255, 0), # 绿
    'mood_prompt': (0, 0, 255),    # 蓝
    'style_prompt': (255, 165, 0)  # 橙
}

# ==========================================
# 2. 纯 NumPy T-SNE 实现 (修复广播错误版)
# ==========================================
def hbeta(D, beta=1.0):
    P = np.exp(-D.copy() * beta)
    sumP = np.sum(P)
    if sumP == 0: sumP = 1e-12
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X, tol=1e-5, perplexity=30.0):
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        Di = np.delete(D[i, :], i)
        betamin = -np.inf
        betamax = np.inf
        H, thisP = hbeta(Di, beta[i])
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.
            H, thisP = hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1
        thisP_full = np.zeros(n)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        thisP_full[mask] = thisP
        P[i, :] = thisP_full
    return P

def tsne_numpy(X, no_dims=2, perplexity=30.0, max_iter=800):
    (n, d) = X.shape
    
    # 自动调整 perplexity 以防止样本过少报错
    if n - 1 < 3 * perplexity:
        perplexity = (n - 1) / 3.0
        print(f"Warning: Perplexity too high for sample size. Adjusted to {perplexity:.2f}")

    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.    # early exaggeration
    P = np.maximum(P, 1e-12)

    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    min_gain = 0.01
    eta = 500
    final_error = 0.0

    print("Running T-SNE iterations...")
    for iter in range(max_iter):
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        PQ = P - Q
        
        # --- 修复 Broadcasting Error ---
        # 错误点：np.tile 生成 (2, N)，而 (Y - Y[i]) 是 (N, 2)
        # 修复方案：利用 broadcasting，计算 (N, 1) * (N, 2) -> (N, 2)
        for i in range(n):
            # scalar_coef: (N,) -> reshaped to (N, 1)
            scalar_coef = (PQ[:, i] * num[:, i])[:, np.newaxis]
            # y_diff: (N, no_dims)
            y_diff = Y[i, :] - Y
            # gradients: (no_dims,)
            dY[i, :] = np.sum(scalar_coef * y_diff, axis=0)
        # --- 修复结束 ---

        if iter < 20: momentum = 0.5
        else: momentum = 0.8

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        if iter == 100: P = P / 4.
        
        if iter % 100 == 0 or iter == max_iter - 1:
            final_error = np.sum(P * np.log(P / Q))
            if iter % 100 == 0:
                print(f"Iteration {iter}: error = {final_error:.4f}")

    return Y, final_error

# ==========================================
# 3. 定量分析模块 (Quantitative Analysis)
# ==========================================
def calculate_cluster_metrics(points, labels):
    unique_labels = list(set(labels))
    stats = {}
    
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        cluster_points = points[indices]
        if len(cluster_points) == 0: continue
        
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        stats[label] = {
            'count': len(cluster_points),
            'centroid': centroid,
            'mean_intra_dist': mean_distance,
            'std_intra_dist': std_distance
        }

    inter_class_dist = {}
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            l1 = unique_labels[i]
            l2 = unique_labels[j]
            if l1 not in stats or l2 not in stats: continue
            
            c1 = stats[l1]['centroid']
            c2 = stats[l2]['centroid']
            dist = np.linalg.norm(c1 - c2)
            inter_class_dist[f"{l1} <-> {l2}"] = dist

    return stats, inter_class_dist

# ==========================================
# 4. 绘图与特征提取
# ==========================================
def save_plot_as_jpg(points, labels, save_path):
    width, height = 1024, 1024
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    
    # 坐标归一化
    margin = 100
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    def normalize(val, v_min, v_max, canvas_size):
        return margin + (val - v_min) / (v_max - v_min + 1e-9) * (canvas_size - 2 * margin)

    # 绘制点
    for i in range(len(points)):
        label = labels[i]
        color = COLORS.get(label, (0, 0, 0))
        px = normalize(x_vals[i], x_min, x_max, width)
        py = normalize(y_vals[i], y_min, y_max, height)
        draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)

    # 绘制图例
    for i, (key, color) in enumerate(COLORS.items()):
        y_pos = 50 + i * 40
        draw.rectangle((width - 250, y_pos, width - 220, y_pos + 30), fill=color)
        draw.text((width - 210, y_pos + 5), key, fill="black")

    img.save(save_path, "JPEG", quality=95)
    print(f"Visualization saved to {save_path}")

def extract_features(model, item_dict):
    with torch.no_grad():
        hint = torch.zeros((1, 3, 512, 512)).cuda().half() if use_fp16 else torch.zeros((1, 3, 512, 512)).cuda()
        
        img_prompts = []
        for key in CONTEXT_PROMPT_KEYS:
            base_prompt = item_dict.get(key, "")
            img_prompts.append(base_prompt + ', best quality, extremely detailed')
            
        c_crossattn = [img_prompts, [""] * 1]
        info = {'glyphs': [], 'gly_line': [], 'positions': [], 'n_lines': [0], 'colors': [], 'masked_x': torch.zeros((1, 4, 64, 64)).cuda().half()}
        
        cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn], text_info=info))
        
        embeddings = {}
        if 'c_crossattn_img_contexts' in cond:
            for key, tensor in cond['c_crossattn_img_contexts'].items():
                # Mean Pooling: [1, 77, 768] -> [768]
                feat = tensor.mean(dim=1).cpu().numpy().flatten()
                embeddings[key] = feat
        return embeddings

def main():
    print("Loading model...")
    model = create_model(config_yaml, use_fp16=use_fp16).cuda().eval()
    if use_fp16: model = model.half()
    model.training_stage = 2
    
    print(f"Loading checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'), strict=False)
    
    print(f"Loading data: {json_path}")
    data_list = load(json_path)
    if isinstance(data_list, dict) and 'data_list' in data_list:
        data_list = data_list['data_list']
    elif not isinstance(data_list, list):
        print("Error: Invalid JSON format. Expected list or dict with 'data_list'.")
        return

    process_num = min(len(data_list), max_items)
    
    all_vectors = []
    all_labels = []
    
    print(f"Extracting features from {process_num} items...")
    for i in tqdm(range(process_num)):
        item = data_list[i]
        # 兼容处理
        default_caption = item.get('caption', '')
        item_dict = {
            'full_prompt': item.get('full_prompt', default_caption),
            'element_prompt': item.get('element_prompt', default_caption),
            'mood_prompt': item.get('mood_prompt', default_caption),
            'style_prompt': item.get('style_prompt', default_caption),
        }
        # 去除占位符
        for k in item_dict: item_dict[k] = item_dict[k].replace('*', ' ')
        
        try:
            feats = extract_features(model, item_dict)
            for k, v in feats.items():
                if k in CONTEXT_PROMPT_KEYS:
                    all_vectors.append(v)
                    all_labels.append(k)
        except Exception as e: 
            print(f"Skipping item {i}: {e}")
            continue

    X = np.array(all_vectors, dtype=np.float64)
    if len(X) == 0:
        print("No features extracted. Exiting.")
        return

    print(f"\nData prepared: {X.shape} (Samples x Dimensions)")
    Y, final_kl_loss = tsne_numpy(X)
    
    stats, inter_dist = calculate_cluster_metrics(Y, all_labels)
    
    print("\n" + "="*50)
    print("DATA FOR THESIS / PAPER ANALYSIS")
    print("="*50)
    print(f"1. T-SNE Parameters:")
    print(f"   - Samples: {len(all_labels)}")
    print(f"   - Original Dimension: {X.shape[1]}")
    print(f"   - Final KL Divergence: {final_kl_loss:.4f}")
    
    print("\n2. Intra-class Compactness (Lower is better):")
    for key, val in stats.items():
        print(f"   - {key:<15}: Mean Dist = {val['mean_intra_dist']:.4f}")
        
    print("\n3. Inter-class Separation (Higher is better):")
    for pair, dist in inter_dist.items():
        print(f"   - {pair:<30}: {dist:.4f}")
    print("="*50 + "\n")

    save_plot_as_jpg(Y, all_labels, output_jpg)

if __name__ == '__main__':
    main()