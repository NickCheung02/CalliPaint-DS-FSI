import os
from PIL import Image, ImageDraw, ImageFont

# ================= 配置区域 =================

# 1. 字体文件路径
FONT_PATH = "/home/610-zzy/Dataset/font/FZQianLXSJW.TTF"

# 2. 图片输出目录 (生成的图片将保存在这里)
OUTPUT_DIR = "/home/610-zzy/Dataset/output_images"

# 3. 宣纸背景颜色 (R, G, B)
# 推荐颜色:
# (250, 249, 222) - 淡米黄 (新宣纸)
# (240, 230, 210) - 仿古黄 (旧宣纸)
PAPER_COLOR = (250, 249, 222) 

# 4. 字体大小
FONT_SIZE = 200

# ===========================================

def ensure_dir(directory):
    """如果目录不存在，则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"已创建输出目录: {directory}")

def create_vertical_calligraphy(text, output_name):
    """生成纵向书法图片"""
    
    # 检查字体
    if not os.path.exists(FONT_PATH):
        print(f"错误: 找不到字体文件 -> {FONT_PATH}")
        return

    try:
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    except Exception as e:
        print(f"字体加载失败: {e}")
        return

    # 布局计算
    padding = int(FONT_SIZE * 0.4)      # 边距
    line_spacing = int(FONT_SIZE * 0.1) # 字间距
    
    # 计算画布尺寸
    canvas_width = FONT_SIZE + (padding * 2)
    canvas_height = (FONT_SIZE * len(text)) + (line_spacing * (len(text) - 1)) + (padding * 2)

    # 创建画布 (使用配置的宣纸颜色)
    image = Image.new('RGB', (canvas_width, canvas_height), color=PAPER_COLOR)
    draw = ImageDraw.Draw(image)

    # 逐字绘制
    current_y = padding
    for char in text:
        # 获取字体的宽和高 (用于居中)
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        
        # 水平居中计算
        x_pos = (canvas_width - char_width) / 2
        
        # 绘制文字 (黑色墨迹)
        draw.text((x_pos, current_y), char, font=font, fill=(20, 20, 20)) # 使用深灰(20,20,20)比纯黑更像墨色
        
        current_y += FONT_SIZE + line_spacing

    # 拼接保存路径
    save_path = os.path.join(OUTPUT_DIR, output_name)
    
    # 保存
    image.save(save_path)
    print(f"[{text}] 已保存至 -> {save_path}")

# --- 主程序执行 ---

if __name__ == "__main__":
    # 1. 确保输出目录存在
    ensure_dir(OUTPUT_DIR)

    # 2. 定义任务列表
    tasks = [
        ("千山鸟飞绝", "poetry_01.png"),
        ("山桃红花在上头", "poetry_02.png")
    ]

    # 3. 批量生成
    for text_content, filename in tasks:
        create_vertical_calligraphy(text_content, filename)