import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import math
import colorsys

# --- 配置参数 ---
input_csv = 'tsne_data.csv'
output_jpg = 'tsne_science_plot.jpg'
font_path = 'font/FZQianLXSJW.TTF'  # 您的项目中自带的字体
img_width = 2400
img_height = 2000
dpi = 300

def parse_args():
    parser = argparse.ArgumentParser(description='Scientific Plotting with PIL (No Matplotlib)')
    parser.add_argument('--input_csv', type=str, default=input_csv)
    parser.add_argument('--output_jpg', type=str, default=output_jpg)
    parser.add_argument('--font_path', type=str, default=font_path)
    parser.add_argument('--plot_mode', type=str, default='text', choices=['text', 'style'], 
                        help="Color by 'text' or 'style'")
    args = parser.parse_args()
    return args

def generate_colors(n):
    """生成 n 个高对比度的科研配色 (类 Seaborn tab10/tab20)"""
    colors = []
    for i in range(n):
        hue = i / n
        # 调整饱和度和亮度，使其看起来不那么刺眼
        saturation = 0.75 
        value = 0.85
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
    return colors

def main():
    args = parse_args()
    
    # 1. 加载数据
    if not os.path.exists(args.input_csv):
        print(f"Error: {args.input_csv} not found.")
        return
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples.")

    # 2. 布局计算 (Layout)
    # 定义绘图区 (Axes) 和 图例区 (Legend)
    margin_top = 150
    margin_bottom = 150
    margin_left = 200
    margin_right = 600 # 右侧留宽一点给图例
    
    plot_w = img_width - margin_left - margin_right
    plot_h = img_height - margin_top - margin_bottom
    
    # 3. 创建画布
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)
    
    # 加载字体
    try:
        title_font = ImageFont.truetype(args.font_path, 80)
        label_font = ImageFont.truetype(args.font_path, 50)
        tick_font = ImageFont.load_default() # 刻度数字用默认字体即可
        # 尝试加载更大的默认字体用于刻度
        try: tick_font = ImageFont.truetype("arial.ttf", 40)
        except: pass
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        tick_font = ImageFont.load_default()

    # 4. 绘制坐标轴和网格 (Axes & Grid)
    # 绘制灰色背景框
    draw.rectangle((margin_left, margin_top, margin_left+plot_w, margin_top+plot_h), 
                   outline='#333333', width=4)
    
    # 绘制网格线 (Grid)
    grid_color = '#E0E0E0'
    grid_steps = 5
    for i in range(1, grid_steps):
        # 垂直网格
        x = margin_left + (plot_w / grid_steps) * i
        draw.line((x, margin_top, x, margin_top+plot_h), fill=grid_color, width=3)
        # 水平网格
        y = margin_top + (plot_h / grid_steps) * i
        draw.line((margin_left, y, margin_left+plot_w, y), fill=grid_color, width=3)

    # 5. 坐标映射
    x_vals = df['tsne_x'].values
    y_vals = df['tsne_y'].values
    
    # 留出 5% 的内边距，防止点压在轴上
    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.05
    x_max += x_range * 0.05
    y_min -= y_range * 0.05
    y_max += y_range * 0.05
    
    def map_coords(x, y):
        # 线性映射到像素坐标
        px = margin_left + (x - x_min) / (x_max - x_min) * plot_w
        # y轴反转 (通常图像坐标系y向下，数学坐标系y向上)
        py = margin_top + (1 - (y - y_min) / (y_max - y_min)) * plot_h 
        return px, py

    # 6. 处理数据分组
    if args.plot_mode == 'text':
        label_col = 'text_content'
        top_n = 15
        title_text = "T-SNE Visualization: Character Semantics"
    else:
        label_col = 'img_name'
        top_n = 12
        title_text = "T-SNE Visualization: Style Source"

    top_labels = df[label_col].value_counts().index[:top_n]
    colors = generate_colors(len(top_labels))
    color_map = {label: colors[i] for i, label in enumerate(top_labels)}
    other_color = '#D3D3D3' # 浅灰色

    # 7. 绘制数据点 (Scatter Plot)
    print("Drawing points...")
    point_radius = 12

    # 先画 'Other'
    other_df = df[~df[label_col].isin(top_labels)]
    for _, row in other_df.iterrows():
        px, py = map_coords(row['tsne_x'], row['tsne_y'])
        draw.ellipse((px-point_radius, py-point_radius, px+point_radius, py+point_radius), 
                     fill=other_color)

    # 再画各类点
    for label in top_labels:
        subset = df[df[label_col] == label]
        c = color_map[label]
        for _, row in subset.iterrows():
            px, py = map_coords(row['tsne_x'], row['tsne_y'])
            # 画点，带一点白色轮廓
            draw.ellipse((px-point_radius-2, py-point_radius-2, px+point_radius+2, py+point_radius+2), 
                         fill='white')
            draw.ellipse((px-point_radius, py-point_radius, px+point_radius, py+point_radius), 
                         fill=c)

    # 8. 绘制中心标注 (仅在 text 模式下)
    if args.plot_mode == 'text':
        for label in top_labels:
            subset = df[df[label_col] == label]
            cx, cy = map_coords(subset['tsne_x'].mean(), subset['tsne_y'].mean())
            
            # 文字背景
            text = str(label)
            bbox = label_font.getbbox(text) # left, top, right, bottom
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            pad = 10
            draw.rectangle((cx-tw/2-pad, cy-th/2-pad, cx+tw/2+pad, cy+th/2+pad), 
                           fill='white', outline='black', width=2)
            # 文字
            draw.text((cx-tw/2, cy-th/2 - 10), text, fill='black', font=label_font)

    # 9. 绘制右侧图例 (Legend)
    print("Drawing legend...")
    legend_x = margin_left + plot_w + 50
    legend_y = margin_top
    
    # 图例标题
    draw.text((legend_x, legend_y), "Legend", fill='black', font=label_font)
    legend_y += 80
    
    # 图例项
    for label in top_labels:
        c = color_map[label]
        # 色块
        draw.rectangle((legend_x, legend_y, legend_x+40, legend_y+40), fill=c, outline='black')
        # 标签文字 (截断过长的文件名)
        label_str = str(label)
        if len(label_str) > 15: label_str = label_str[:12] + "..."
        
        draw.text((legend_x + 60, legend_y - 5), label_str, fill='black', font=label_font)
        legend_y += 70

    # 'Other' 图例
    draw.rectangle((legend_x, legend_y, legend_x+40, legend_y+40), fill=other_color, outline='black')
    draw.text((legend_x + 60, legend_y - 5), "Other", fill='gray', font=label_font)

    # 10. 绘制标题和轴标签
    # 标题居中
    title_w = title_font.getlength(title_text)
    draw.text(((img_width - title_w)/2, 50), title_text, fill='black', font=title_font)
    
    # 轴标签
    draw.text(((img_width)/2 - 100, img_height - 100), "Dimension 1", fill='black', font=label_font)
    # Y轴标签 (需要旋转，PIL旋转文字比较麻烦，这里简化为水平写在左上角)
    draw.text((30, margin_top - 60), "Dim 2", fill='black', font=label_font)

    # 11. 保存
    image.save(args.output_jpg, quality=95, dpi=(dpi, dpi))
    print(f"Scientific plot saved to: {args.output_jpg}")

if __name__ == '__main__':
    main()