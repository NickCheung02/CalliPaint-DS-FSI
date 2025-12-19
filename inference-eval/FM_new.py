import sys
import os
import cv2
import einops
import numpy as np
import torch
import random
import time
from PIL import ImageFont
from tqdm import tqdm

# --- 引入项目依赖 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from t3_dataset import draw_glyph, draw_glyph2, get_text_caption
from dataset_util import load

# ================= 配置区域 (请确认路径) =================
config_yaml = './models_yaml/anytext2_sd15.yaml'
ckpt_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/checkpoints/lightning_logs/version_1/checkpoints/epoch=24-step=900.ckpt'
# 请确保这里指向正确的测试集 JSON 路径
json_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/test-result/poem_feature.json' 
output_dir = './gradcam_data_storage' # 数据保存目录
font_path = '/home/610-zzy/AnyText2-main-Real0922-DoubleStage-FHS-4/font/FZQianLXSJW.TTF'

# 其他参数
num_samples = 1 
image_resolution = 512
strength = 1.0
ddim_steps = 20
scale = 7.5
seed = 100
eta = 0.0
a_prompt = 'best quality, extremely detailed'
n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, watermark'
PLACE_HOLDER = '*'
max_chars = 20
max_lines = 20
glyph_scale = 1
use_fp16 = True
default_color = [500, 500, 500]
fonthint_type = 'Arial'
CONTEXT_PROMPT_KEYS = ['full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']

# 尝试加载字体
try:
    font = ImageFont.truetype(font_path, size=60)
except Exception as e:
    print(f"Warning: Could not load font from {font_path}. Using default.", e)
    font = ImageFont.load_default()

# ================= Grad-CAM 类定义 =================
class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook

        def save_gradient(name):
            def hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return hook

        for name, layer in self.target_layers.items():
            self.hooks.append(layer.register_forward_hook(save_activation(name)))
            self.hooks.append(layer.register_full_backward_hook(save_gradient(name)))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def generate_cam(self, target_name):
        grad = self.gradients[target_name].float()
        act = self.activations[target_name].float()
        
        # Global Average Pooling (GAP)
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        # Weighted combination
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        # ReLU
        cam = torch.relu(cam)
        
        # Normalize (Min-Max)
        # 注意：这里我们保存归一化后的数据，方便后续直接可视化
        # 也可以选择不归一化保存原始值，取决于后续需求
        if torch.max(cam) > 0:
            cam = cam - torch.min(cam)
            cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.squeeze().cpu().numpy()

# ================= 辅助函数 =================
def arr2tensor(arr, bs):
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))
    _arr = torch.from_numpy(arr.copy()).float().cuda()
    if use_fp16:
        _arr = _arr.half()
    _arr = torch.stack([_arr for _ in range(bs)], dim=0)
    return _arr

def draw_pos(ploygon, prob=1.0):
    img = np.zeros((512, 512, 1))
    if random.random() < prob:
        pts = ploygon.reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color=255)
    return img/255.

# ================= 数据生成逻辑 =================
def generate_gradcam_data(model, ddim_sampler, item_dict, save_dir):
    # 1. 准备数据
    with torch.no_grad():
        n_lines = item_dict['n_lines']
        pos_imgs = item_dict['positions']
        glyphs = item_dict['glyphs']
        gly_line = item_dict['gly_line']
        colors = item_dict['color']
        hint = np.sum(pos_imgs, axis=0).clip(0, 1)
        H, W, = (512, 512)
        
        if seed == -1:
            curr_seed = random.randint(0, 65535)
        else:
            curr_seed = seed
        seed_everything(curr_seed)

        info = {}
        info['glyphs'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [n_lines]*num_samples
        info['colors'] = colors
        for i in range(n_lines):
            glyph = glyphs[i]
            pos = pos_imgs[i]
            gline = gly_line[i]
            info['glyphs'] += [arr2tensor(glyph, num_samples)]
            info['gly_line'] += [arr2tensor(gline, num_samples)]
            info['positions'] += [arr2tensor(pos, num_samples)]
            info['colors'][i] = arr2tensor(info['colors'][i], num_samples)/255.

        ref_img = np.zeros((H, W, 3))
        masked_img = ((ref_img.astype(np.float32) / 127.5) - 1.0 - hint*10).clip(-1, 1)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cuda()
        if use_fp16:
            masked_img = masked_img.half()
        encoder_posterior = model.encode_first_stage(masked_img[None, ...])
        masked_x = model.get_first_stage_encoding(encoder_posterior).detach()
        if use_fp16:
            masked_x = masked_x.half()
        info['masked_x'] = torch.cat([masked_x for _ in range(num_samples)], dim=0)

        hint = arr2tensor(hint, num_samples)
        info['font_hint'] = arr2tensor(item_dict['font_hint'], num_samples)

        # Conditioning
        text_prompt_cond = item_dict['text_caption']
        img_prompts_flat_list_cond = []
        for key in CONTEXT_PROMPT_KEYS:
            prompt = item_dict.get(key, item_dict['img_caption']) 
            full_img_prompt = prompt + ', ' + a_prompt
            img_prompts_flat_list_cond.extend([full_img_prompt] * num_samples)
        
        text_prompts_list_cond = [text_prompt_cond] * num_samples
        c_crossattn_cond = [img_prompts_flat_list_cond, text_prompts_list_cond]
        cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn_cond], text_info=info))

        text_prompt_uncond = ''
        img_prompts_flat_list_uncond = []
        for _ in CONTEXT_PROMPT_KEYS:
            img_prompts_flat_list_uncond.extend([n_prompt] * num_samples)
            
        text_prompts_list_uncond = [text_prompt_uncond] * num_samples
        c_crossattn_uncond = [img_prompts_flat_list_uncond, text_prompts_list_uncond]
        un_cond = model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[c_crossattn_uncond], text_info=info))
        
        shape = (4, H // 8, W // 8)
        model.control_scales = ([strength] * 13)

    # 2. 推理并捕获轨迹
    print(f"Running inference to capture trajectory for: {item_dict['img_name']}")
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                 shape, cond, verbose=False, eta=eta,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=un_cond,
                                                 log_every_t=1) 
    
    latents_trajectory = intermediates['x_inter'] 
    timesteps_trajectory = intermediates.get('index', list(reversed(range(0, 1000, 1000//ddim_steps))))

    # 3. Grad-CAM 计算与数据收集
    target_layers = {
        "middle_block": model.model.diffusion_model.middle_block,
        "output_block_8": model.model.diffusion_model.output_blocks[8]
    }
    grad_cam = GradCAM(model, target_layers)

    # 选择关键时间步 (Start, Middle, End)
    total_steps = len(latents_trajectory)
    # 我们选择大约 0%, 50%, 95% 进度的点，您可以根据需要增加更多点
    indices_to_analyze = [0, total_steps // 2, total_steps - 2] 
    
    # 存储字典
    data_to_save = {}
    
    print(f"\n--- Extracting Feature Map Data ---")
    for idx in indices_to_analyze:
        if idx >= len(latents_trajectory): continue
        
        x_t = latents_trajectory[idx].detach()
        t_val = timesteps_trajectory[idx]
        
        x_t.requires_grad = True
        t_tensor = torch.full((x_t.shape[0],), t_val, device=x_t.device, dtype=torch.long)
        
        model.zero_grad()
        model_output = model.apply_model(x_t, t_tensor, cond)
        loss = model_output.norm()
        loss.backward()
        
        for layer_name in target_layers.keys():
            cam_data = grad_cam.generate_cam(layer_name)
            
            # 构造存储键名：layer_name_timestep
            # 也可以使用多层字典，但 npz 扁平化存储更简单
            key = f"{layer_name}_t{t_val}"
            data_to_save[key] = cam_data
            
            # 打印数据统计信息
            print(f"[Data Captured] T={t_val} | Layer={layer_name} | Shape={cam_data.shape} | Range=[{cam_data.min():.4f}, {cam_data.max():.4f}]")
            
    grad_cam.remove_hooks()
    
    # 保存原始图片 (作为参考，不画图，只存解码后的像素矩阵)
    if use_fp16:
        samples = samples.half()
    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    data_to_save["final_image"] = x_samples[0]
    data_to_save["indices"] = np.array(indices_to_analyze)
    data_to_save["timesteps"] = np.array([timesteps_trajectory[i] for i in indices_to_analyze])
    
    # 执行保存
    save_path = os.path.join(save_dir, f"gradcam_data_{item_dict['img_name'].split('.')[0]}.npz")
    np.savez_compressed(save_path, **data_to_save)
    print(f"\n[Success] All data saved to: {save_path}")
    print("Keys in stored file:", list(data_to_save.keys()))


# ================= 数据加载 wrappers (保持不变) =================
def load_data_wrapper(input_path):
    content = load(input_path)
    d = []
    data_list = []
    if isinstance(content, dict) and 'data_list' in content:
        data_list = content['data_list']
    elif isinstance(content, list):
        data_list = content
    else:
        return []

    for gt in data_list:
        info = {}
        info['img_name'] = gt['img_name']
        default_caption = gt.get('caption', '')
        info['caption'] = default_caption
        info['full_prompt'] = gt.get('full_prompt', default_caption)
        info['element_prompt'] = gt.get('element_prompt', default_caption)
        info['mood_prompt'] = gt.get('mood_prompt', default_caption)
        info['style_prompt'] = gt.get('style_prompt', default_caption)

        if PLACE_HOLDER in info['caption']:
            for k in ['caption', 'full_prompt', 'element_prompt', 'mood_prompt', 'style_prompt']:
                info[k] = info[k].replace(PLACE_HOLDER, " ")
            
        if 'annotations' in gt:
            polygons = []
            texts = []
            pos = []
            for annotation in gt['annotations']:
                if len(annotation['polygon']) == 0: continue
                if 'valid' in annotation and annotation['valid'] is False: continue
                polygons.append(annotation['polygon'])
                texts.append(annotation['text'])
                if 'pos' in annotation: pos.append(annotation['pos'])
            if len(texts) == 0:
                texts = [' ', ]
                polygons = [[[0, 0], [0, 50], [50, 50], [50, 0]], ]
                pos = [0, ]
            info['polygons'] = [np.array(i) for i in polygons]
            info['texts'] = texts
            info['pos'] = pos
        d.append(info)
    return d

def get_item_wrapper(data_list, item):
    item_dict = {}
    cur_item = data_list[item]
    item_dict['img_name'] = cur_item['img_name']
    item_dict['img_caption'] = cur_item['caption']
    item_dict['full_prompt'] = cur_item['full_prompt']
    item_dict['element_prompt'] = cur_item['element_prompt']
    item_dict['mood_prompt'] = cur_item['mood_prompt']
    item_dict['style_prompt'] = cur_item['style_prompt']
    item_dict['text_caption'] = ''
    item_dict['glyphs'] = []
    item_dict['gly_line'] = []
    item_dict['positions'] = []
    item_dict['texts'] = []
    item_dict['color'] = []
    texts = cur_item.get('texts', [])
    if len(texts) > 0:
        sel_idxs = [i for i in range(len(texts))]
        if len(texts) > max_lines:
            sel_idxs = sel_idxs[:max_lines]
        item_dict['text_caption'] = get_text_caption(len(sel_idxs), PLACE_HOLDER)
        item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
        item_dict['texts'] = [cur_item['texts'][i][:max_chars] for i in sel_idxs]
        item_dict['color'] += [np.array(default_color)] * len(sel_idxs)
        for idx, text in enumerate(item_dict['texts']):
            gly_line = draw_glyph(font, text)
            glyphs = draw_glyph2(font, text, item_dict['polygons'][idx], item_dict['color'][idx], scale=glyph_scale)
            item_dict['glyphs'] += [glyphs]
            item_dict['gly_line'] += [gly_line]
        for polygon in item_dict['polygons']:
            item_dict['positions'] += [draw_pos(polygon, 1.0)]
            
    n_lines = min(len(texts), max_lines)
    item_dict['n_lines'] = n_lines
    n_pad = max_lines - n_lines
    if n_pad > 0:
        item_dict['glyphs'] += [np.zeros((512*glyph_scale, 512*glyph_scale, 3))] * n_pad
        item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
        item_dict['positions'] += [np.zeros((512, 512, 1))] * n_pad
        item_dict['color'] += [np.array(default_color)] * n_pad
        
    if fonthint_type == 'Arial':
        item_dict['font_hint'] = cv2.resize(np.sum(np.stack(item_dict['glyphs']), axis=0).clip(0, 1), (512, 512))
    else:
        item_dict['font_hint'] = np.zeros((512, 512, 3))
    return item_dict

if __name__ == '__main__':
    print("--- Feature Map Evolution Data Generation ---")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)
    
    data_list = load_data_wrapper(json_path)
    print(f"Loaded {len(data_list)} items.")
    
    model = create_model(config_yaml, use_fp16=use_fp16).cuda().eval()
    if use_fp16:
        model = model.half()
    model.training_stage = 2 
    model.load_state_dict(load_state_dict(ckpt_path, location='cuda'), strict=False)
    ddim_sampler = DDIMSampler(model)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认处理第一张图片，如需处理多张请改为循环
    target_idx = 0 
    item_dict = get_item_wrapper(data_list, target_idx)
    
    generate_gradcam_data(model, ddim_sampler, item_dict, output_dir)