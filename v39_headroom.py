import cv2
import numpy as np
import os
import sys
import time
import onnxruntime as ort
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter

# ==========================================
# UI 绘图工具库 (Pro HUD)
# ==========================================

def draw_hud_label(img, text, x, y, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (t_w, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    cv2.rectangle(img, (x, y - t_h - pad), (x + t_w + pad*2, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return x + t_w + pad*2 + 10

def draw_bracket(img, x1, y1, x2, y2, color, thickness=2, length=20):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2 + length, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 15, 1)

def draw_exclusion_zone(img, limit_x, side="RIGHT", h=1080):
    overlay = img.copy()
    w = img.shape[1]
    color = (255, 0, 0) # Blue for Shield
    
    if side == "RIGHT":
        cv2.rectangle(overlay, (int(limit_x), 0), (w, h), color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.line(img, (int(limit_x), 0), (int(limit_x), h), color, 3)
    elif side == "LEFT":
        cv2.rectangle(overlay, (0, 0), (int(limit_x), h), color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.line(img, (int(limit_x), 0), (int(limit_x), h), color, 3)

# ==========================================
# 工具类：颜色指纹锁定 (Identity Lock)
# ==========================================
class IdentityTracker:
    def __init__(self):
        self.target_hist = None
        self.locked = False
        
    def get_histogram(self, img_hsv, mask=None):
        hist = cv2.calcHist([img_hsv], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
        
    def lock_target(self, img_bgr, rect):
        x, y, w, h = rect
        roi = img_bgr[y:y+h, x:x+w]
        if roi.size == 0: return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        self.target_hist = self.get_histogram(hsv)
        self.locked = True
        print("   [Tracker] 主体特征已锁定 (Color Fingerprint).")
        
    def compare(self, img_bgr, rect):
        if not self.locked or self.target_hist is None: return 0.0
        x, y, w, h = rect
        roi = img_bgr[y:y+h, x:x+w]
        if roi.size == 0: return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = self.get_histogram(hsv)
        score = cv2.compareHist(self.target_hist, hist, cv2.HISTCMP_CORREL)
        return score

# ==========================================
# 深度学习显著性检测器 (U2Net)
# ==========================================
class U2NetSaliency:
    def __init__(self, model_path='u2netp.onnx'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:] 
        
    def detect(self, img_bgr):
        h, w = self.input_shape
        img_resized = cv2.resize(img_bgr, (w, h))
        img_norm = img_resized.astype(np.float32) / 255.0
        tmp_img = np.zeros((h, w, 3), dtype=np.float32)
        tmp_img[:, :, 0] = (img_norm[:, :, 0] - 0.485) / 0.229
        tmp_img[:, :, 1] = (img_norm[:, :, 1] - 0.456) / 0.224
        tmp_img[:, :, 2] = (img_norm[:, :, 2] - 0.406) / 0.225
        tmp_img = tmp_img.transpose((2, 0, 1))
        tmp_img = np.expand_dims(tmp_img, 0)
        inputs = {self.input_name: tmp_img}
        outputs = self.session.run(None, inputs)
        d1 = outputs[0][0, 0, :, :]
        ma, mi = np.max(d1), np.min(d1)
        d1 = (d1 - mi) / (ma - mi + 1e-8)
        mask = (d1 * 255).astype(np.uint8)
        return mask

# ==========================================
# 核心算法引擎 V39.0 (Full Frame Entry & Headroom)
# ==========================================

class DirectorEngine:
    def __init__(self, target_res=(1920, 1080)):
        print(f"[System] 初始化导演引擎 V39.0 (Smart Headroom & Fast Entry)...")
        self.target_w, self.target_h = target_res
        self.aspect_ratio = self.target_w / self.target_h
        self.tracker = IdentityTracker()
        try:
            self.saliency_detector = U2NetSaliency('u2netp.onnx')
            print("   [Model] U2NetP Active.")
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            sys.exit(1)

    def smooth_array_strong(self, data, window_size):
        """ 强力平滑：包含 V38 的安全检查 """
        data_len = len(data)
        if data_len < 3: return data
        if window_size >= data_len: window_size = data_len
        if window_size % 2 == 0: window_size -= 1
        if window_size < 3: return data
        try:
            return savgol_filter(data, window_size, 2)
        except ValueError:
            return data

    def get_scene_analysis(self, img, frame_idx):
        img_w, img_h = img.shape[1], img.shape[0]
        
        # 1. Detect
        saliency_map_small = self.saliency_detector.detect(img)
        saliency_map = cv2.resize(saliency_map_small, (img_w, img_h))
        
        # 2. Threshold
        _, thresh = cv2.threshold(saliency_map, 100, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.erode(thresh, kernel, iterations=2) 
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_objs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < (img_w * img_h * 0.003): continue 
            x, y, w, h = cv2.boundingRect(c)
            valid_objs.append({'rect': (x,y,w,h), 'area': area, 'cnt': c})

        # --- Identity Lock ---
        if frame_idx == 0 and not self.tracker.locked and valid_objs:
            best_candidate = None
            max_score = -1
            for obj in valid_objs:
                x, y, w, h = obj['rect']
                aspect = h / w
                cx = x + w/2
                dist_center = 1.0 - abs(cx - img_w/2)/(img_w/2)
                score = (aspect if aspect > 1.0 else 0.5) * dist_center * obj['area']
                if score > max_score:
                    max_score = score
                    best_candidate = obj
            if best_candidate:
                self.tracker.lock_target(img, best_candidate['rect'])

        # --- Match Subject ---
        subject = None
        aliens = []
        best_match_score = -1
        
        for obj in valid_objs:
            sim = self.tracker.compare(img, obj['rect'])
            if sim > best_match_score:
                best_match_score = sim
                subject = obj
        
        if best_match_score < 0.2: subject = None

        if subject:
            for obj in valid_objs:
                if obj is not subject: aliens.append(obj)
        else:
            aliens = valid_objs

        # --- Wall Detection ---
        raw_limit_l = 0
        raw_limit_r = img_w
        
        if subject:
            sx, sy, sw, sh = subject['rect']
            s_cx = sx + sw/2
            
            for alien in aliens:
                ax, ay, aw, ah = alien['rect']
                is_edge = (ax < img_w * 0.1) or ((ax + aw) > img_w * 0.9)
                if (ax + aw) < s_cx:
                    if (ax + aw) > raw_limit_l: raw_limit_l = ax + aw
                elif ax > s_cx:
                    if ax < raw_limit_r: raw_limit_r = ax

            buffer = img_w * 0.02
            raw_limit_l = min(raw_limit_l + buffer, sx)
            raw_limit_r = max(raw_limit_r - buffer, sx + sw)

        return {
            'subject_rect': subject['rect'] if subject else None,
            'limit_l': raw_limit_l,
            'limit_r': raw_limit_r,
            'saliency': saliency_map
        }

    def plan_cinematic_path(self, raw_data_list, src_w, src_h):
        num_frames = len(raw_data_list)
        if num_frames == 0: return [], [], []
        
        # 1. 提取序列
        raw_cx = []
        raw_cy = []
        raw_sh = [] # 需要人物高度来计算 headroom
        raw_l = []
        raw_r = []
        
        last_valid_s = (src_w/2, src_h/2, 100, src_h/3)
        for d in raw_data_list:
            if d['subject_rect']:
                last_valid_s = d['subject_rect']
            
            sx, sy, sw, sh = last_valid_s
            raw_cx.append(sx + sw/2)
            raw_cy.append(sy + sh/2) # 原始中心
            raw_sh.append(sh)        # 记录高度
            raw_l.append(d['limit_l'])
            raw_r.append(d['limit_r'])
            
        raw_cx = np.array(raw_cx)
        raw_l = np.array(raw_l)
        raw_r = np.array(raw_r)
        raw_sh = np.array(raw_sh)

        # 2. 墙壁稳定化 (Wall Stabilization)
        stable_l = np.zeros_like(raw_l)
        stable_r = np.zeros_like(raw_r)
        curr_l, curr_r = 0.0, float(src_w)
        
        for i in range(num_frames):
            if raw_l[i] > curr_l: curr_l = raw_l[i]
            else: curr_l -= 1.0
            if curr_l < 0: curr_l = 0
            stable_l[i] = curr_l
            
            if raw_r[i] < curr_r: curr_r = raw_r[i]
            else: curr_r += 1.0
            if curr_r > src_w: curr_r = src_w
            stable_r[i] = curr_r
            
        win_wall = min(num_frames, 48)
        stable_l = self.smooth_array_strong(stable_l, win_wall)
        stable_r = self.smooth_array_strong(stable_r, win_wall)
        
        # 3. 目标安全宽度计算 (Target Safe Width)
        # 计算全局最窄瓶颈
        required_widths = []
        for i in range(num_frames):
            dist_l = raw_cx[i] - stable_l[i]
            dist_r = stable_r[i] - raw_cx[i]
            min_side = min(dist_l, dist_r)
            if min_side < 50: min_side = 50 
            req_w = min_side * 2.0
            required_widths.append(req_w)
            
        # 找到一个安全的、可持续的构图宽度
        # 取 10% 分位数，确保绝大多数时候干扰物都在外面
        safe_cruise_width = np.percentile(required_widths, 10)
        safe_cruise_width = min(safe_cruise_width, src_w * 0.8) # 至少缩放一点
        safe_cruise_width = max(safe_cruise_width, 300) # 别缩太小
        
        # 4. 生成“快速入场”变焦轨迹 (Fast Entry Zoom)
        # 需求：Frame 0 = src_w, Frame 36 (1.5s) = safe_cruise_width
        transition_frames = min(int(36), num_frames)
        
        target_w_seq = np.full(num_frames, safe_cruise_width)
        
        # 创建线性插值 (Ken Burns Zoom In)
        entry_ramp = np.linspace(src_w, safe_cruise_width, transition_frames)
        target_w_seq[:transition_frames] = entry_ramp
        
        # 5. 生成中心点轨迹 (Virtual Gimbal)
        # 需求：Frame 0 = Image Center, Frame 36 = Subject Center
        
        # 目标中心点序列 (Target Center Sequence)
        # 我们希望最终稳定在 Subject Center (raw_cx)，但要经过平滑
        win_cam = min(num_frames, 72)
        ideal_cx_smooth = self.smooth_array_strong(raw_cx, win_cam)
        
        # 同样，前 N 帧需要从画面中心 (src_w/2) 过渡到 ideal_cx_smooth
        final_cx = []
        start_cx_val = src_w / 2.0
        
        for i in range(num_frames):
            # 计算当前时刻的“理想”平滑中心
            target_c = ideal_cx_smooth[i]
            
            # 如果在过渡期内，进行插值
            if i < transition_frames:
                t = i / float(transition_frames)
                # Ease out cubic for smoothness
                t = 1 - pow(1 - t, 3) 
                current_cx = start_cx_val * (1-t) + target_c * t
            else:
                current_cx = target_c
            
            # 硬约束检查 (Collision Check)
            curr_w = target_w_seq[i]
            half_w = curr_w / 2.0
            l, r = stable_l[i], stable_r[i]
            
            # 优先避让墙壁
            if (current_cx - half_w) < l: current_cx = l + half_w 
            elif (current_cx + half_w) > r: current_cx = r - half_w 
            
            final_cx.append(current_cx)
            
        final_cx = np.array(final_cx)
        
        # 6. 垂直构图修正 (Headroom Correction)
        # 目标：取景框上边缘应该在头顶上方 (Height * 0.15) 处
        # Head Top = raw_cy (center) - raw_sh/2
        final_cy = []
        
        # 平滑高度数据
        smooth_sh = self.smooth_array_strong(raw_sh, win_cam)
        # 平滑 Y 坐标
        raw_cy_smooth = self.smooth_array_strong(np.array(raw_cy), win_cam)
        
        start_cy_val = src_h / 2.0

        for i in range(num_frames):
            w = target_w_seq[i]
            h = w / self.aspect_ratio
            
            # 计算人物头顶位置
            head_top = raw_cy_smooth[i] - smooth_sh[i]/2
            
            # 期望的取景框上边缘 (Top Margin)
            # 留出 20% 的画面高度作为头顶空间
            desired_box_top = head_top - (h * 0.20)
            
            # 推导出 Box Center Y
            target_c_y = desired_box_top + h / 2
            
            # 入场过渡
            if i < transition_frames:
                t = i / float(transition_frames)
                t = 1 - pow(1 - t, 3)
                current_cy = start_cy_val * (1-t) + target_c_y * t
            else:
                current_cy = target_c_y
                
            final_cy.append(current_cy)
            
        final_cy = np.array(final_cy)

        # 7. 最终生成 Rects
        final_rects = []
        for i in range(num_frames):
            cx = final_cx[i]
            cy = final_cy[i]
            w = target_w_seq[i]
            h = w / self.aspect_ratio
            
            # 屏幕边界强制钳制
            if w > src_w: w = src_w; h = w / self.aspect_ratio
            if (cx - w/2) < 0: cx = w/2
            if (cx + w/2) > src_w: cx = src_w - w/2
            if (cy - h/2) < 0: cy = h/2
            if (cy + h/2) > src_h: cy = src_h - h/2
            
            final_rects.append((cx, cy, w, h))
            
        return {
            'boxes': final_rects,
            'l_wall': stable_l,
            'r_wall': stable_r
        }

    def analyze_content(self, clip):
        print(f"   [Phase 1] 逐帧扫描 (Raw Scan)...")
        src_w, src_h = clip.w, clip.h
        
        raw_data_list = []
        step = 2 
        
        for i, frame in enumerate(clip.iter_frames()):
            if i % step == 0:
                small_frame = cv2.resize(frame, (640, 360))
                analysis = self.get_scene_analysis(small_frame, i)
                scale = src_w / 640.0
                
                s_rect_scaled = None
                if analysis['subject_rect']:
                    sx, sy, sw, sh = analysis['subject_rect']
                    s_rect_scaled = (sx*scale, sy*scale, sw*scale, sh*scale)
                
                raw_data_list.append({
                    'subject_rect': s_rect_scaled,
                    'limit_l': analysis['limit_l'] * scale,
                    'limit_r': analysis['limit_r'] * scale,
                    'saliency': analysis['saliency']
                })
            else:
                raw_data_list.append(None)
        
        for k in range(len(raw_data_list)):
            if raw_data_list[k] is None:
                prev = k - 1
                while prev >= 0 and raw_data_list[prev] is None: prev -= 1
                if prev >= 0: raw_data_list[k] = raw_data_list[prev]
                else:
                    succ = k + 1
                    while succ < len(raw_data_list) and raw_data_list[succ] is None: succ += 1
                    if succ < len(raw_data_list): raw_data_list[k] = raw_data_list[succ]

        print(f"   [Phase 2] 全局路径规划 (Fast Entry & Headroom)...")
        final_plan = self.plan_cinematic_path(raw_data_list, src_w, src_h)
        final_plan['saliency'] = [d['saliency'] for d in raw_data_list]
        return final_plan

    def render_hud_monitor(self, clip, data):
        boxes = data["boxes"]
        l_wall = data["l_wall"]
        r_wall = data["r_wall"]
        saliency_maps = data["saliency"]
        
        src_w, src_h = clip.w, clip.h
        
        def frame_process(get_frame, t):
            raw_frame = get_frame(t)
            idx = min(int(t * clip.fps), len(boxes)-1)
            
            # BG
            s_map = saliency_maps[idx]
            if s_map.shape[:2] != (src_h, src_w):
                s_map = cv2.resize(s_map, (src_w, src_h))
            heatmap = cv2.applyColorMap(s_map, cv2.COLORMAP_JET)
            monitor_bg = cv2.addWeighted(raw_frame, 0.7, heatmap, 0.3, 0)
            
            # Walls
            rl = r_wall[idx]
            ll = l_wall[idx]
            if rl < src_w * 0.99: draw_exclusion_zone(monitor_bg, rl, "RIGHT", src_h)
            if ll > src_w * 0.01: draw_exclusion_zone(monitor_bg, ll, "LEFT", src_h)
            
            # Box
            cx, cy, cw, ch = boxes[idx]
            ix1, iy1 = int(cx - cw/2), int(cy - ch/2)
            ix2, iy2 = int(cx + cw/2), int(cy + ch/2)
            
            cv2.rectangle(monitor_bg, (ix1, iy1), (ix2, iy2), (0, 255, 0), 3)
            draw_bracket(monitor_bg, ix1, iy1, ix2, iy2, (0, 255, 0), 3, 40)
            
            # Preview
            rx1 = max(0, ix1); ry1 = max(0, iy1)
            rx2 = min(src_w, ix2); ry2 = min(src_h, iy2)
            if rx2 <= rx1 or ry2 <= ry1: rx1,ry1,rx2,ry2 = 0,0,src_w,src_h
            
            crop_img = raw_frame[ry1:ry2, rx1:rx2]
            try:
                final_res_img = cv2.resize(crop_img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            except:
                final_res_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            
            scale = self.target_h / src_h
            mon_w = int(src_w * scale)
            mon_img = cv2.resize(monitor_bg, (mon_w, self.target_h))
            
            combined = np.zeros((self.target_h, self.target_w * 2, 3), dtype=np.uint8)
            off_x = (self.target_w - mon_w) // 2
            combined[:, off_x:off_x+mon_w] = mon_img
            combined[:, self.target_w:] = final_res_img
            
            draw_hud_label(combined, f"ZOOM: ENTRY", 20, self.target_h - 100, bg_color=(0,100,0))
            if ll > 10 or rl < src_w - 10:
                draw_hud_label(combined, f"WALL: ACTIVE", 20, self.target_h - 60, bg_color=(0,0,200))
            
            return combined

        return clip.fl(frame_process)

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    engine = DirectorEngine()
    
    VIDEO_PLAYLIST = [
        r"C:\Users\Administrator\Desktop\vlog-clean\data\6.mp4"
    ]
    
    print("\n" + "="*60)
    print("【AI 导演监视器 V39.0 - Headroom & Fast Entry】")
    print("="*60 + "\n")
    
    if not os.path.exists("u2netp.onnx"):
        print("!!! 缺少 u2netp.onnx 模型")
        sys.exit(1)
    
    for i, video_path in enumerate(VIDEO_PLAYLIST):
        print(f">>> [{i+1}/{len(VIDEO_PLAYLIST)}] 正在处理: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"!!! 警告: 文件不存在，跳过")
            continue
            
        try:
            clip = VideoFileClip(video_path)
            data = engine.analyze_content(clip)
            hud_clip = engine.render_hud_monitor(clip, data)
            
            out_dir = "TRACKING/V39_HEADROOM"
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            
            fname = os.path.splitext(os.path.basename(video_path))[0]
            out_name = os.path.join(out_dir, f"HUD_V39_{fname}_Headroom.mp4")
            
            print(f"    正在渲染至: {out_name} ...")
            hud_clip.write_videofile(out_name, codec='libx264', audio_codec='aac', fps=24, threads=4, logger='bar')
            
            clip.close()
            hud_clip.close()
            print("√ 完成\n")
            
        except Exception as e:
            print(f"!!! 处理出错: {e}")
            import traceback
            traceback.print_exc()
            
    print("所有任务处理完毕！")