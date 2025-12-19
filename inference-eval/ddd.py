import numpy as np
import os

# ================= 配置 =================
# 你的 .npz 文件路径
npz_path = "gradcam_data_storage/gradcam_data_shanshui_001.npz"
# =======================================

def calculate_stats(heatmap):
    """计算热力图的关键统计指标"""
    # 归一化 (防止不同层级量纲不同)
    if heatmap.max() > heatmap.min():
        norm_map = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        norm_map = heatmap

    # 1. Mean Activation (整体激活强度)
    mean_val = np.mean(norm_map)
    
    # 2. Peak Activation (峰值强度 - 原始值)
    peak_val = np.max(heatmap)
    
    # 3. Sparsity / Focus (聚焦度/稀疏度)
    # 简单的做法是看有多少像素超过了 0.5 的阈值。值越小说明越聚焦。
    focus_ratio = np.sum(norm_map > 0.5) / norm_map.size
    
    # 4. Variance (纹理丰富度/对比度)
    variance = np.var(norm_map)
    
    return mean_val, peak_val, focus_ratio, variance

def analyze_data():
    if not os.path.exists(npz_path):
        print("找不到文件")
        return

    data = np.load(npz_path)
    timesteps = sorted(data['timesteps'], reverse=True) # 确保从噪声(T=999)到成图(T=0)
    
    # 获取层名称（通过解析 key）
    keys = list(data.keys())
    layers = set()
    for k in keys:
        if "_t" in k:
            layer_name = k.split("_t")[0]
            layers.add(layer_name)
            
    print("=== 请复制以下内容发送给我 ===")
    print(f"File: {os.path.basename(npz_path)}")
    print(f"TimeSteps Checked: {timesteps}")
    print("-" * 40)

    for layer in layers:
        print(f"\n[Layer Analysis: {layer}]")
        print(f"{'TimeStep':<10} | {'Peak(原始)':<12} | {'Focus(聚焦度)':<15} | {'Var(对比度)':<12}")
        print("-" * 55)
        
        # 按时间顺序遍历 (T大 -> T小)
        for t in timesteps:
            key = f"{layer}_t{t}"
            if key in data:
                heatmap = data[key]
                mean_v, peak_v, focus_r, var_v = calculate_stats(heatmap)
                
                # 聚焦度：数字越小，说明热力越集中在某些点（比如笔画），而不是散在背景里
                print(f"{t:<10} | {peak_v:.4f}       | {focus_r:.4f}          | {var_v:.4f}")
            else:
                print(f"{t:<10} | N/A")
                
    print("\n" + "="*40)

if __name__ == "__main__":
    analyze_data()