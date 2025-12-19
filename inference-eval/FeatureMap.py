# import sys
# import os
# import cv2
# import einops
# import numpy as np
# import torch
# import random
# import time
# import argparse
# import matplotlib.pyplot as plt
# from PIL import ImageFont
# from tqdm import tqdm

# # 添加上级目录以导入项目模块
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from pytorch_lightning import seed_everything
# from cldm.model import create_model, load_state_dict
# from cldm.ddim_hacked import DDIMSampler
# from t3_dataset import draw_glyph, draw_glyph2, get_text_caption
# from dataset_util import load

# # ==========================================
# # 1. Configuration (所有参数设置在此)
# # ==========================================
# class Config:
#     # 路径设置 (请根据您的实际环境修改)
#     config_yaml = './models_yaml/anytext2_sd15.yaml'
#     ckpt_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/checkpoints/lightning_logs/version_1/checkpoints/epoch=24-step=900.ckpt'
#     input_json = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/test-result/poem_feature.json' 
#     output_dir = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/environment/feature_analysis_v1' # 结果输出目录
#     font_path = 'font/FZQianLXSJW.TTF' # 字体路径
    
#     # 生成参数
#     num_samples = 1   # 分析时建议设为 1，避免生成过多数据
#     image_resolution = 512
#     strength = 1.0
#     ddim_steps = 20
#     scale = 7.5
#     seed = 100
#     eta = 0.0
    
#     # Prompt 设置
#     a_prompt = 'best quality, extremely detailed'
#     n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark'
#     context_prompt_keys = ['full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']
    
#     # 文本渲染参数
#     max_chars = 20
#     max_lines = 20
#     glyph_scale = 1
#     use_fp16 = True
#     default_color = [500, 500, 500]
#     fonthint_type = 'Arial'
#     place_holder = '*'
    
#     # 系统参数
#     save_memory = False

# # 初始化全局配置
# cfg = Config()

# # ==========================================
# # 2. Feature Map Analysis Tool
# # ==========================================
# class FeatureMapVisualizer:
#     def __init__(self, model, save_dir):
#         """
#         初始化特征图可视化器
#         """
#         # 访问内部的 UNet 模型
#         self.unet = model.model.diffusion_model 
#         self.save_dir = save_dir
#         self.activations = {}
#         self.hooks = []
        
#         # 定义需要监控的关键层 (Input -> Middle -> Output)
#         # 这里的命名编号 (01, 02...) 用于保证打印和绘图时的顺序
#         self.layer_mapping = {
#             '01_Input_Block_1': self.unet.input_blocks[1],
#             '02_Input_Block_4': self.unet.input_blocks[4], # Downsample
#             '03_Input_Block_8': self.unet.input_blocks[8], # Downsample
#             '04_Middle_Block':  self.unet.middle_block,    # Bottleneck
#             '05_Output_Block_3': self.unet.output_blocks[3], # Upsample
#             '06_Output_Block_6': self.unet.output_blocks[6], # Upsample
#             '07_Output_Block_9': self.unet.output_blocks[9], 
#             '08_Output_Block_11': self.unet.output_blocks[11] # Output
#         }

#     def _hook_fn(self, name):
#         def hook(module, input, output):
#             # output shape: [batch, channels, h, w]
#             # 我们只取 batch 中的第一个样本，并 detach 到 CPU
#             if isinstance(output, tuple):
#                 out_tensor = output[0]
#             else:
#                 out_tensor = output
#             self.activations[name] = out_tensor[0].detach().cpu()
#         return hook

#     def register_hooks(self):
#         """注册 Forward Hooks"""
#         self.hooks = []
#         for name, layer in self.layer_mapping.items():
#             hook = layer.register_forward_hook(self._hook_fn(name))
#             self.hooks.append(hook)
#         # print("DEBUG: Feature Map Hooks registered.")

#     def remove_hooks(self):
#         """移除 Hooks，防止内存泄漏"""
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks = []
#         self.activations = {}
#         # print("DEBUG: Feature Map Hooks removed.")

#     def analyze_statistics(self, img_name):
#         """
#         打印每一层的统计数据，用于论文 Quantitative Analysis
#         """
#         print(f"\n{'='*30} Feature Map Statistics: {img_name} {'='*30}")
#         print(f"{'Layer Name':<25} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}")
#         print(f"{'-'*85}")
        
#         sorted_keys = sorted(self.activations.keys())
#         stats_list = []
        
#         for name in sorted_keys:
#             feat = self.activations[name].float()
#             mean_val = feat.mean().item()
#             std_val = feat.std().item()
#             min_val = feat.min().item()
#             max_val = feat.max().item()
            
#             print(f"{name:<25} | {mean_val:.4f}     | {std_val:.4f}     | {min_val:.4f}     | {max_val:.4f}")
#             stats_list.append((name, mean_val, std_val, min_val, max_val))
            
#         print(f"{'='*85}\n")
#         return stats_list

#     def save_heatmaps(self, prefix_name):
#         """
#         生成并保存热力图演变图
#         """
#         if not self.activations:
#             print("Warning: No activations captured.")
#             return

#         save_path = os.path.join(self.save_dir, 'feature_maps_analysis', prefix_name)
#         os.makedirs(save_path, exist_ok=True)
        
#         # 打印统计数据供分析
#         self.analyze_statistics(prefix_name)
        
#         plt.figure(figsize=(24, 4))
#         sorted_keys = sorted(self.activations.keys())
#         num_layers = len(sorted_keys)

#         for i, name in enumerate(sorted_keys):
#             feat = self.activations[name] # [C, H, W]
            
#             # 1. 沿通道维度求平均 (Mean Pooling) -> [H, W]
#             heatmap = torch.mean(feat, dim=0).numpy()
            
#             # 2. 归一化 (Min-Max Normalization)
#             heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
#             # 3. 统一尺寸 (512x512) 以便视觉对比
#             heatmap_resized = cv2.resize(heatmap, (512, 512), interpolation=cv2.INTER_NEAREST)
            
#             # 4. 绘图
#             ax = plt.subplot(1, num_layers, i + 1)
#             ax.imshow(heatmap_resized, cmap='viridis') # 使用 Viridis 配色，论文常用
#             ax.set_title(name.split('_', 1)[1], fontsize=9)
#             ax.axis('off')

#         plt.tight_layout()
#         save_file = os.path.join(save_path, f'{prefix_name}_evolution.jpg')
#         plt.savefig(save_file, dpi=150)
#         plt.close()
#         print(f"[Analysis] Evolution map saved to: {save_file}")

# # ==========================================
# # 3. Helper Functions
# # ==========================================
# def arr2tensor(arr, bs):
#     if len(arr.shape) == 3:
#         arr = np.transpose(arr, (2, 0, 1))
#     _arr = torch.from_numpy(arr.copy()).float().cuda()
#     if cfg.use_fp16:
#         _arr = _arr.half()
#     _arr = torch.stack([_arr for _ in range(bs)], dim=0)
#     return _arr

# def draw_pos(ploygon, prob=1.0):
#     img = np.zeros((512, 512, 1))
#     if random.random() < prob:
#         pts = ploygon.reshape((-1, 1, 2))
#         cv2.fillPoly(img, [pts], color=255)
#     return img/255.

# def load_data_wrapper(input_path):
#     """加载数据并处理格式差异"""
#     if not os.path.exists(input_path):
#         print(f"Error: Input file not found: {input_path}")
#         return []
        
#     content = load(input_path)
#     d = []
#     count = 0
    
#     data_list = []
#     if isinstance(content, dict) and 'data_list' in content:
#         data_list = content['data_list']
#     elif isinstance(content, list):
#         data_list = content
#     else:
#         print(f"Error: Unexpected JSON format.")
#         return []

#     for gt in data_list:
#         info = {}
#         info['img_name'] = gt['img_name']
        
#         default_caption = gt.get('caption', '')
#         info['caption'] = default_caption
#         info['full_prompt'] = gt.get('full_prompt', default_caption)
#         info['element_prompt'] = gt.get('element_prompt', default_caption)
#         info['mood_prompt'] = gt.get('mood_prompt', default_caption)
#         info['style_prompt'] = gt.get('style_prompt', default_caption)

#         # 处理 Placeholder
#         for key in ['caption', 'full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']:
#             if cfg.place_holder in info[key]:
#                 info[key] = info[key].replace(cfg.place_holder, " ")
#                 if key == 'caption': count += 1

#         if 'annotations' in gt:
#             polygons = []
#             texts = []
#             pos = []
#             for annotation in gt['annotations']:
#                 if len(annotation['polygon']) == 0: continue
#                 if 'valid' in annotation and not annotation['valid']: continue
#                 polygons.append(annotation['polygon'])
#                 texts.append(annotation['text'])
#                 if 'pos' in annotation: pos.append(annotation['pos'])
            
#             if len(texts) == 0:
#                 texts = [' ', ]
#                 polygons = [[[0, 0], [0, 50], [50, 50], [50, 0]], ]
#                 pos = [0, ]
#             info['polygons'] = [np.array(i) for i in polygons]
#             info['texts'] = texts
#             info['pos'] = pos
#         d.append(info)
    
#     print(f'{input_path} loaded, imgs={len(d)}')
#     return d

# def get_item_wrapper(data_list, item, font):
#     item_dict = {}
#     cur_item = data_list[item]
#     item_dict['img_name'] = cur_item['img_name']
    
#     for key in ['img_caption', 'full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']:
#         # 映射 key 名称
#         src_key = 'caption' if key == 'img_caption' else key
#         item_dict[key] = cur_item.get(src_key, cur_item.get('caption', ''))

#     item_dict['text_caption'] = ''
#     item_dict['glyphs'] = []
#     item_dict['gly_line'] = []
#     item_dict['positions'] = []
#     item_dict['texts'] = []
#     item_dict['color'] = []
    
#     texts = cur_item.get('texts', [])
#     if len(texts) > 0:
#         sel_idxs = [i for i in range(len(texts))]
#         if len(texts) > cfg.max_lines:
#             sel_idxs = sel_idxs[:cfg.max_lines]
        
#         item_dict['text_caption'] = get_text_caption(len(sel_idxs), cfg.place_holder)
#         item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
#         item_dict['texts'] = [cur_item['texts'][i][:cfg.max_chars] for i in sel_idxs]
#         item_dict['color'] += [np.array(cfg.default_color)] * len(sel_idxs)
        
#         for idx, text in enumerate(item_dict['texts']):
#             gly_line = draw_glyph(font, text)
#             glyphs = draw_glyph2(font, text, item_dict['polygons'][idx], item_dict['color'][idx], scale=cfg.glyph_scale)
#             item_dict['glyphs'] += [glyphs]
#             item_dict['gly_line'] += [gly_line]
        
#         for polygon in item_dict['polygons']:
#             item_dict['positions'] += [draw_pos(polygon, 1.0)]

#     # Padding
#     n_lines = min(len(texts), cfg.max_lines)
#     item_dict['n_lines'] = n_lines
#     n_pad = cfg.max_lines - n_lines
#     if n_pad > 0:
#         item_dict['glyphs'] += [np.zeros((512*cfg.glyph_scale, 512*cfg.glyph_scale, 3))] * n_pad
#         item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
#         item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
#         item_dict['color'] += [np.array(cfg.default_color)] * n_pad

#     # Font Hint
#     if cfg.fonthint_type == 'Arial':
#         item_dict['font_hint'] = cv2.resize(np.sum(np.stack(item_dict['glyphs']), axis=0).clip(0, 1), (512, 512))
#     else:
#         item_dict['font_hint'] = np.zeros((512, 512, 3))
        
#     return item_dict

# # ==========================================
# # 4. Core Process Function (With Analysis)
# # ==========================================
# def process(model, ddim_sampler, item_dict):
#     with torch.no_grad():
#         n_lines = item_dict['n_lines']
#         pos_imgs = item_dict['positions']
#         glyphs = item_dict['glyphs']
#         gly_line = item_dict['gly_line']
#         colors = item_dict['color']
#         hint = np.sum(pos_imgs, axis=0).clip(0, 1)
#         H, W = (cfg.image_resolution, cfg.image_resolution)
        
#         current_seed = cfg.seed
#         if current_seed == -1:
#             current_seed = random.randint(0, 65535)
#         seed_everything(current_seed)
        
#         if cfg.save_memory:
#             model.low_vram_shift(is_diffusing=False)

#         # Batch construction
#         bs = cfg.num_samples
#         info = {}
#         info['glyphs'] = []
#         info['gly_line'] = []
#         info['positions'] = []
#         info['n_lines'] = [n_lines] * bs
#         info['colors'] = colors
        
#         for i in range(n_lines):
#             info['glyphs'] += [arr2tensor(glyphs[i], bs)]
#             info['gly_line'] += [arr2tensor(gly_line[i], bs)]
#             info['positions'] += [arr2tensor(pos_imgs[i], bs)]
#             info['colors'][i] = arr2tensor(info['colors'][i], bs)/255.
            
#         # Masked Image
#         ref_img = np.zeros((H, W, 3))
#         masked_img = ((ref_img.astype(np.float32) / 127.5) - 1.0 - hint*10).clip(-1, 1)
#         masked_img = np.transpose(masked_img, (2, 0, 1))
#         masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
#         if cfg.use_fp16: masked_img = masked_img.half()
        
#         encoder_posterior = model.encode_first_stage(masked_img[None, ...])
#         masked_x = model.get_first_stage_encoding(encoder_posterior).detach()
#         if cfg.use_fp16: masked_x = masked_x.half()
#         info['masked_x'] = torch.cat([masked_x for _ in range(bs)], dim=0)

#         hint = arr2tensor(hint, bs)
#         info['font_hint'] = arr2tensor(item_dict['font_hint'], bs)

#         # ---------------- Conditioning ----------------
#         text_prompt_cond = item_dict['text_caption']
#         img_prompts_cond = []
#         for key in cfg.context_prompt_keys:
#             prompt = item_dict.get(key, item_dict['img_caption'])
#             full_prompt = prompt + ', ' + cfg.a_prompt
#             img_prompts_cond.extend([full_prompt] * bs)
            
#         c_crossattn_cond = [img_prompts_cond, [text_prompt_cond] * bs]
#         cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn_cond], text_info=info))

#         # ---------------- Unconditioning ----------------
#         img_prompts_uncond = []
#         for _ in cfg.context_prompt_keys:
#             img_prompts_uncond.extend([cfg.n_prompt] * bs)
            
#         c_crossattn_uncond = [img_prompts_uncond, [''] * bs]
#         un_cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn_uncond], text_info=info))

#         # ---------------- Sampling & Analysis ----------------
#         shape = (4, H // 8, W // 8)
#         if cfg.save_memory:
#             model.low_vram_shift(is_diffusing=True)
#         model.control_scales = ([cfg.strength] * 13)

#         # >>>>>> Analysis Hook Start >>>>>>
#         visualizer = None
#         try:
#             # 自动使用 cfg.output_dir
#             visualizer = FeatureMapVisualizer(model, cfg.output_dir)
#             visualizer.register_hooks()
#         except Exception as e:
#             print(f"Analysis Error: Could not init visualizer: {e}")
#         # <<<<<< Analysis Hook End <<<<<<

#         tic = time.time()
#         samples, intermediates = ddim_sampler.sample(
#             cfg.ddim_steps, bs, shape, cond, verbose=False, eta=cfg.eta,
#             unconditional_guidance_scale=cfg.scale,
#             unconditional_conditioning=un_cond
#         )
#         cost = (time.time() - tic)*1000.

#         # >>>>>> Analysis Save Start >>>>>>
#         if visualizer:
#             img_prefix = item_dict['img_name'].split('.')[0]
#             # 这里会自动打印统计数据 (Mean/Std/Min/Max)
#             visualizer.save_heatmaps(img_prefix) 
#             visualizer.remove_hooks()
#         # <<<<<< Analysis Save End <<<<<<

#         if cfg.save_memory:
#             model.low_vram_shift(is_diffusing=False)
            
#         if cfg.use_fp16: samples = samples.half()
#         x_samples = model.decode_first_stage(samples)
#         x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

#         results = [x_samples[i] for i in range(bs)]
#         results += [cost]
#         return results

# # ==========================================
# # 5. Main Execution
# # ==========================================
# def parse_args():
#     parser = argparse.ArgumentParser(description='AnyText2 Inference & Analysis')
#     # 允许命令行覆盖 Config 中的部分设置
#     parser.add_argument('--input_json', type=str, default=cfg.input_json)
#     parser.add_argument('--output_dir', type=str, default=cfg.output_dir)
#     return parser.parse_args()

# if __name__ == '__main__':
#     args = parse_args()
    
#     # 更新 Config
#     cfg.input_json = args.input_json
#     cfg.output_dir = args.output_dir
    
#     # 加载字体
#     try:
#         font = ImageFont.truetype(cfg.font_path, size=60)
#     except Exception as e:
#         print(f"Font Load Error: {e}. Trying Arial...")
#         font = ImageFont.truetype('arial.ttf', size=60)

#     # 加载数据
#     data_list = load_data_wrapper(cfg.input_json)
#     if not data_list:
#         sys.exit(1)

#     # 加载模型
#     print(f"Loading model from {cfg.ckpt_path}...")
#     model = create_model(cfg.config_yaml, use_fp16=cfg.use_fp16).cuda().eval()
#     if cfg.use_fp16: model = model.half()
#     model.training_stage = 2
#     model.load_state_dict(load_state_dict(cfg.ckpt_path, location='cuda'), strict=False)
#     ddim_sampler = DDIMSampler(model)

#     os.makedirs(cfg.output_dir, exist_ok=True)

#     times = []
#     print("Start Inference & Analysis...")
    
#     for i in tqdm(range(len(data_list)), desc='Generator'):
#         item_dict = get_item_wrapper(data_list, i, font)
        
#         # 检查是否已存在
#         img_name_base = item_dict['img_name'].split('.')[0]
#         # 简单检查最后一张样本是否存在
#         last_sample_name = f"{img_name_base}_{cfg.num_samples-1}.jpg"
#         if os.path.exists(os.path.join(cfg.output_dir, last_sample_name)):
#             continue

#         try:
#             results = process(model, ddim_sampler, item_dict)
#             times.append(results.pop()) # 取出时间
            
#             for idx, img in enumerate(results):
#                 save_name = f"{img_name_base}_{idx}.jpg"
#                 cv2.imwrite(os.path.join(cfg.output_dir, save_name), img[..., ::-1])
                
#         except Exception as e:
#             print(f"Error processing {item_dict['img_name']}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue

#     if times:
#         print(f'Mean Time: {np.mean(times)/1000.:.2f} s.')
#     print(f"All done. Feature maps saved to {os.path.join(cfg.output_dir, 'feature_maps_analysis')}")


import sys
import os
import cv2
import einops
import numpy as np
import torch
import random
import time
import argparse
from PIL import ImageFont
from tqdm import tqdm

# 添加上级目录以导入项目模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from t3_dataset import draw_glyph, draw_glyph2, get_text_caption
from dataset_util import load

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # 路径设置
    config_yaml = './models_yaml/anytext2_sd15.yaml'
    ckpt_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/checkpoints/lightning_logs/version_1/checkpoints/epoch=24-step=900.ckpt'
    input_json = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/test-result/poem_feature.json' 
    output_dir = './test-result/feature_data_v1' # 数据保存目录
    font_path = 'font/FZQianLXSJW.TTF'
    
    # 生成参数
    num_samples = 1
    image_resolution = 512
    strength = 1.0
    ddim_steps = 20
    scale = 7.5
    seed = 100
    eta = 0.0
    
    # Prompt & Context
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark'
    context_prompt_keys = ['full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']
    
    # Text Rendering
    max_chars = 20
    max_lines = 20
    glyph_scale = 1
    use_fp16 = True
    default_color = [500, 500, 500]
    fonthint_type = 'Arial'
    place_holder = '*'
    
    # System
    save_memory = False

cfg = Config()

# ==========================================
# 2. Feature Map Recorder (No Matplotlib)
# ==========================================
class FeatureMapRecorder:
    def __init__(self, model, save_dir):
        """
        初始化特征图记录器
        """
        self.unet = model.model.diffusion_model 
        self.save_dir = save_dir
        self.activations = {}
        self.hooks = []
        
        # 建立数据保存目录
        self.data_save_dir = os.path.join(save_dir, 'feature_data_arrays')
        self.stats_save_dir = os.path.join(save_dir, 'feature_statistics_logs')
        os.makedirs(self.data_save_dir, exist_ok=True)
        os.makedirs(self.stats_save_dir, exist_ok=True)
        
        # 定义监控层级
        self.layer_mapping = {
            '01_Input_Block_1': self.unet.input_blocks[1],
            '02_Input_Block_4': self.unet.input_blocks[4],
            '03_Input_Block_8': self.unet.input_blocks[8],
            '04_Middle_Block':  self.unet.middle_block,
            '05_Output_Block_3': self.unet.output_blocks[3],
            '06_Output_Block_6': self.unet.output_blocks[6],
            '07_Output_Block_9': self.unet.output_blocks[9], 
            '08_Output_Block_11': self.unet.output_blocks[11]
        }

    def _hook_fn(self, name):
        def hook(module, input, output):
            # 兼容 tuple 输出 (某些层输出可能是 tuple)
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            # 只取 Batch 的第一个样本，detach 防止显存泄漏
            self.activations[name] = out_tensor[0].detach().cpu()
        return hook

    def register_hooks(self):
        self.hooks = []
        for name, layer in self.layer_mapping.items():
            hook = layer.register_forward_hook(self._hook_fn(name))
            self.hooks.append(hook)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    def save_data(self, prefix_name):
        """
        处理流程：
        1. 计算统计数据 (Mean, Std, Min, Max) 并打印/保存为 TXT
        2. 将特征图沿通道平均 (Mean Pooling)，转为 numpy
        3. 保存所有层的数据为 .npz 文件
        """
        if not self.activations:
            print("Warning: No activations captured.")
            return

        # 1. 准备统计日志文件
        stats_file_path = os.path.join(self.stats_save_dir, f"{prefix_name}_stats.txt")
        stats_buffer = []
        
        header = f"{'Layer Name':<25} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10}"
        div_line = "-" * 85
        
        print(f"\n{'='*30} Feature Statistics: {prefix_name} {'='*30}")
        print(header)
        print(div_line)
        
        stats_buffer.append(f"Image: {prefix_name}")
        stats_buffer.append(header)
        stats_buffer.append(div_line)

        # 2. 准备数据字典
        data_to_save = {}
        sorted_keys = sorted(self.activations.keys())

        for name in sorted_keys:
            feat = self.activations[name].float() # [C, H, W]
            
            # --- 统计分析 ---
            mean_val = feat.mean().item()
            std_val = feat.std().item()
            min_val = feat.min().item()
            max_val = feat.max().item()
            
            log_line = f"{name:<25} | {mean_val:.4f}     | {std_val:.4f}     | {min_val:.4f}     | {max_val:.4f}"
            print(log_line)
            stats_buffer.append(log_line)
            
            # --- 数据处理 ---
            # 为了节省空间且方便画热力图，我们先在通道维度求平均 [C, H, W] -> [H, W]
            # 如果您以后需要分析特定通道，可以去掉 mean(0)，直接保存 feat.numpy()
            heatmap_2d = torch.mean(feat, dim=0).numpy()
            data_to_save[name] = heatmap_2d

        print(f"{'='*85}\n")
        
        # 3. 保存统计 TXT
        with open(stats_file_path, 'w') as f:
            f.write('\n'.join(stats_buffer))
        
        # 4. 保存特征数据 NPZ
        npz_file_path = os.path.join(self.data_save_dir, f"{prefix_name}_features.npz")
        np.savez_compressed(npz_file_path, **data_to_save)
        
        print(f"[Recorder] Data saved to: {npz_file_path}")
        print(f"[Recorder] Stats saved to: {stats_file_path}")

# ==========================================
# 3. Helper Functions
# ==========================================
def arr2tensor(arr, bs):
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    if cfg.use_fp16: _arr = _arr.half()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr

def draw_pos(ploygon, prob=1.0):
    img = np.zeros((512, 512, 1))
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    return img/255.

def load_data_wrapper(input_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return []
    content = load(input_path)
    data_list = content['data_list'] if isinstance(content, dict) and 'data_list' in content else content
    if not isinstance(data_list, list): return []

    d = []
    count = 0
    for gt in data_list:
        info = {}
        info['img_name'] = gt['img_name']
        default_caption = gt.get('caption', '')
        info['caption'] = default_caption
        # 填充 prompt 字段
        for k in cfg.context_prompt_keys:
            info[k] = gt.get(k, gt.get(k.replace('_prompt', '_caption'), default_caption)) # 尝试不同命名兼容

        # 清理 placeholder
        for key in ['caption'] + cfg.context_prompt_keys:
            if key in info and cfg.place_holder in info[key]:
                info[key] = info[key].replace(cfg.place_holder, " ")
                if key == 'caption': count += 1

        if 'annotations' in gt:
            polygons, texts, pos = [], [], []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0: continue
                if 'valid' in annotation and not annotation['valid']: continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                if 'pos' in annotation: pos.append(annotation['pos'])
            
            if not texts:
                texts = [' ']
                polygons = [[[0, 0], [0, 50], [50, 50], [50, 0]]]
                pos = [0]
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    print(f'{input_path} loaded, imgs={len(d)}')
    return d

def get_item_wrapper(data_list, item, font):
    cur_item = data_list[item]
    item_dict = {'img_name': cur_item['img_name']}
    # 统一 prompt 字段
    for key in ['img_caption'] + cfg.context_prompt_keys:
        src_key = 'caption' if key == 'img_caption' else key
        item_dict[key] = cur_item.get(src_key, cur_item.get('caption', ''))

    item_dict.update({'text_caption': '', 'glyphs': [], 'gly_line': [], 'positions': [], 'texts': [], 'color': []})
    
    texts = cur_item.get('texts', [])
    if texts:
        sel_idxs = list(range(min(len(texts), cfg.max_lines)))
        item_dict['text_caption'] = get_text_caption(len(sel_idxs), cfg.place_holder)
        item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
        item_dict['texts'] = [cur_item['texts'][i][:cfg.max_chars] for i in sel_idxs]
        item_dict['color'] += [np.array(cfg.default_color)] * len(sel_idxs)
        
        for idx, text in enumerate(item_dict['texts']):
            item_dict['gly_line'].append(draw_glyph(font, text))
            item_dict['glyphs'].append(draw_glyph2(font, text, item_dict['polygons'][idx], item_dict['color'][idx], scale=cfg.glyph_scale))
            item_dict['positions'].append(draw_pos(item_dict['polygons'][idx]))

    # Padding
    n_pad = cfg.max_lines - min(len(texts), cfg.max_lines)
    if n_pad > 0:
        item_dict['glyphs'] += [np.zeros((512*cfg.glyph_scale, 512*cfg.glyph_scale, 3))] * n_pad
        item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
        item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
        item_dict['color'] += [np.array(cfg.default_color)] * n_pad
    
    item_dict['n_lines'] = min(len(texts), cfg.max_lines)
    # Font Hint
    if cfg.fonthint_type == 'Arial':
        item_dict['font_hint'] = cv2.resize(np.sum(np.stack(item_dict['glyphs']), axis=0).clip(0, 1), (512, 512))
    else:
        item_dict['font_hint'] = np.zeros((512, 512, 3))
    return item_dict

# ==========================================
# 4. Process Function
# ==========================================
def process(model, ddim_sampler, item_dict):
    with torch.no_grad():
        n_lines = item_dict['n_lines']
        bs = cfg.num_samples
        H, W = cfg.image_resolution, cfg.image_resolution
        
        seed_everything(cfg.seed if cfg.seed != -1 else random.randint(0, 65535))
        if cfg.save_memory: model.low_vram_shift(False)

        # Batch Construction
        info = {
            'glyphs': [], 'gly_line': [], 'positions': [], 
            'n_lines': [n_lines]*bs, 'colors': item_dict['color']
        }
        for i in range(n_lines):
            info['glyphs'].append(arr2tensor(item_dict['glyphs'][i], bs))
            info['gly_line'].append(arr2tensor(item_dict['gly_line'][i], bs))
            info['positions'].append(arr2tensor(item_dict['positions'][i], bs))
            info['colors'][i] = arr2tensor(info['colors'][i], bs)/255.

        # Masked Image
        hint = np.sum(item_dict['positions'], axis=0).clip(0, 1)
        masked_img = ((np.zeros((H, W, 3), dtype=np.float32)/127.5)-1.0 - hint*10).clip(-1, 1)
        masked_img = torch.from_numpy(np.transpose(masked_img, (2, 0, 1))).float().cuda()
        if cfg.use_fp16: masked_img = masked_img.half()
        
        masked_x = model.get_first_stage_encoding(model.encode_first_stage(masked_img[None, ...])).detach()
        if cfg.use_fp16: masked_x = masked_x.half()
        info['masked_x'] = torch.cat([masked_x]*bs, dim=0)
        
        hint_tensor = arr2tensor(hint, bs)
        info['font_hint'] = arr2tensor(item_dict['font_hint'], bs)

        # Prompts
        cond_prompts, uncond_prompts = [], []
        for key in cfg.context_prompt_keys:
            p = item_dict.get(key, item_dict['img_caption'])
            cond_prompts.extend([p + ', ' + cfg.a_prompt] * bs)
            uncond_prompts.extend([cfg.n_prompt] * bs)
            
        cond = model.get_learned_conditioning(dict(
            c_concat=[hint_tensor], 
            c_crossattn=[[cond_prompts, [item_dict['text_caption']]*bs]], 
            text_info=info
        ))
        un_cond = model.get_learned_conditioning(dict(
            c_concat=[hint_tensor], 
            c_crossattn=[[uncond_prompts, ['']*bs]], 
            text_info=info
        ))

        # Sampling
        shape = (4, H // 8, W // 8)
        if cfg.save_memory: model.low_vram_shift(True)
        model.control_scales = ([cfg.strength] * 13)

        # >>>>>> Recorder Init >>>>>>
        recorder = None
        try:
            recorder = FeatureMapRecorder(model, cfg.output_dir)
            recorder.register_hooks()
        except Exception as e:
            print(f"Recorder Error: {e}")
        # <<<<<< Recorder Init <<<<<<

        tic = time.time()
        samples, _ = ddim_sampler.sample(
            cfg.ddim_steps, bs, shape, cond, verbose=False, eta=cfg.eta,
            unconditional_guidance_scale=cfg.scale,
            unconditional_conditioning=un_cond
        )
        cost = (time.time() - tic)*1000.

        # >>>>>> Recorder Save >>>>>>
        if recorder:
            img_prefix = item_dict['img_name'].split('.')[0]
            recorder.save_data(img_prefix) # 保存数据和统计信息
            recorder.remove_hooks()
        # <<<<<< Recorder Save <<<<<<

        if cfg.save_memory: model.low_vram_shift(False)
        if cfg.use_fp16: samples = samples.half()
        
        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(bs)]
        results += [cost]
        return results

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default=cfg.input_json)
    parser.add_argument('--output_dir', type=str, default=cfg.output_dir)
    args = parser.parse_args()
    
    cfg.input_json = args.input_json
    cfg.output_dir = args.output_dir
    os.makedirs(cfg.output_dir, exist_ok=True)

    try:
        font = ImageFont.truetype(cfg.font_path, size=60)
    except:
        font = ImageFont.truetype('arial.ttf', size=60)

    data_list = load_data_wrapper(cfg.input_json)
    if not data_list: sys.exit(1)

    print(f"Loading model from {cfg.ckpt_path}...")
    model = create_model(cfg.config_yaml, use_fp16=cfg.use_fp16).cuda().eval()
    if cfg.use_fp16: model = model.half()
    model.training_stage = 2
    model.load_state_dict(load_state_dict(cfg.ckpt_path, location='cuda'), strict=False)
    ddim_sampler = DDIMSampler(model)

    times = []
    print("Start Inference & Recording...")
    
    for i in tqdm(range(len(data_list))):
        item = get_item_wrapper(data_list, i, font)
        try:
            results = process(model, ddim_sampler, item)
            times.append(results.pop())
            for idx, img in enumerate(results):
                name = f"{item['img_name'].split('.')[0]}_{idx}.jpg"
                cv2.imwrite(os.path.join(cfg.output_dir, name), img[..., ::-1])
        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()

    if times: print(f'Mean Time: {np.mean(times)/1000.:.2f} s.')
    print(f"Done. Data saved to {cfg.output_dir}")