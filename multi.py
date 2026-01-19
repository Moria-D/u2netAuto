import cv2
import numpy as np
import os
import sys
import time
import json
import onnxruntime as ort
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter

# ==========================================
# 工具类：JSON 日志记录器 (修复 NumPy 2.0 兼容性)
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    """
    专门用于处理 NumPy 数据类型的 JSON 编码器
    [V60.2 修复] 使用抽象基类替代具体类型，解决 NumPy 2.0 移除 np.float_ 导致的 AttributeError
    """
    def default(self, obj):
        # 处理所有 NumPy 整数类型 (int8, int16, int32, int64, uint8 等)
        if isinstance(obj, np.integer):
            return int(obj)
        # 处理所有 NumPy 浮点类型 (float16, float32, float64 等)
        # 注意：不再使用 np.float_，因为它在 NumPy 2.0 中已被移除
        elif isinstance(obj, np.floating):
            return float(obj)
        # 处理 NumPy 数组
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理 NumPy 布尔值
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # 兜底处理
        return super().default(obj)

class EventLogger:
    def __init__(self):
        self.logs = {
            "video_file": "",
            "process_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events": []
        }

    def set_file(self, filename):
        self.logs["video_file"] = filename

    def log_event(self, event_type, start_time, end_time, params):
        """
        记录运镜事件
        :param event_type: 'ZOOM', 'PAN', 'TRACKING', 'KEN_BURNS'
        :param start_time: 事件开始时间 (秒)
        :param end_time: 事件结束时间 (秒)
        :param params: 字典格式的参数 (如 scale, target_x, target_y)
        """
        entry = {
            "type": event_type,
            "timestamp_start": float(f"{start_time:.2f}"),
            "timestamp_end": float(f"{end_time:.2f}"),
            "duration": float(f"{end_time - start_time:.2f}"),
            "parameters": params
        }
        self.logs["events"].append(entry)
        # 实时打印
        print(f"   [LOG] {event_type} | T: {start_time:.1f}s-{end_time:.1f}s | Params: {params}")

    def save_to_disk(self, filepath):
        # 使用修复后的 NumpyEncoder
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"   [System] 日志已保存至: {filepath}")

# ==========================================
# UI 绘图工具库 (完整保留)
# ==========================================
def draw_hud_label(img, text, x, y, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (t_w, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    cv2.rectangle(img, (x, y - t_h - pad), (x + t_w + pad*2, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return x + t_w + pad*2 + 10

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1):
    points = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
    for i in range(4):
        p1 = points[i]; p2 = points[(i+1)%4]
        cv2.line(img, p1, p2, color, thickness)

def draw_vector_arrow(img, p1, p2, color=(0, 255, 255), thickness=4, tip_size=0.5):
    p1 = (int(p1[0]), int(p1[1])); p2 = (int(p2[0]), int(p2[1]))
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if dist > 2.0: 
        cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=tip_size)

def draw_bracket(img, x1, y1, x2, y2, color, thickness=2, length=20):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_progress_bar(img, x, y, w, h, val, max_val, color=(0, 255, 0), label=""):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), -1)
    ratio = max(0.0, min(1.0, val / max_val))
    fill_w = int(w * ratio)
    cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 1)
    if label:
        cv2.putText(img, f"{label}: {val:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# ==========================================
# 深度学习显著性检测器 (U2Net)
# ==========================================
class U2NetSaliency:
    def __init__(self, model_path='u2netp.onnx'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型文件: {model_path}\n请下载 u2netp.onnx 并放入目录")
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
# 核心算法引擎 V60.2 (NumPy 2.0 Compatible Ultimate)
# ==========================================

class DirectorEngine:
    def __init__(self, target_res=(1920, 1080)):
        print(f"[System] 初始化导演引擎 V60.2 (Tracking + KP Fusion)...")
        self.target_w, self.target_h = target_res
        self.logger = EventLogger()
        try:
            self.saliency_detector = U2NetSaliency('u2netp.onnx')
            print("   [Model] U2NetP Active.")
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            sys.exit(1)

    # --- 通用数学辅助函数 ---
    def smooth_data(self, data, window_size, polyorder=3):
        data_len = len(data)
        if data_len < 4: return data 
        if window_size >= data_len: window_size = data_len - 1
        if window_size % 2 == 0: window_size -= 1
        if window_size < 3: return data
        if polyorder >= window_size: polyorder = window_size - 1
        return savgol_filter(data, window_size, polyorder)

    def ease_quintic_in_out(self, t): # V32 Tracking curve
        if t < 0.5: return 16 * t * t * t * t * t
        else: return 1 - pow(-2 * t + 2, 5) / 2

    def ease_out_cubic(self, t): # V56 Zoom start curve
        return 1 - pow(1 - t, 3)

    def ease_sine_in_out(self, t): # V56 Pan curve
        return -(np.cos(np.pi * t) - 1) / 2

    def map_range(self, val, in_min, in_max, out_min, out_max):
        if val <= in_min: return out_min
        if val >= in_max: return out_max
        return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def get_saliency_roi(self, img):
        saliency_map_small = self.saliency_detector.detect(img)
        saliency_map = cv2.resize(saliency_map_small, (img.shape[1], img.shape[0]))
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        _, thresh = cv2.threshold(saliency_map, 100, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        saliency_ratio = white_pixels / (img.shape[0] * img.shape[1])
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        final_cx, final_cy = img.shape[1]/2, img.shape[0]/2
        sal_top, sal_bottom = 0, img.shape[0]
        rect = (0, 0, img.shape[1], img.shape[0])
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            rx, ry, rw, rh = cv2.boundingRect(main_contour)
            rect = (rx, ry, rw, rh)
            sal_top = ry
            sal_bottom = ry + rh
            
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                phys_cx = int(M["m10"] / M["m00"])
                phys_cy = int(M["m01"] / M["m00"])
            else:
                phys_cx, phys_cy = rx + rw/2, ry + rh/2
                
            final_cx, final_cy = phys_cx, phys_cy
                
        return (final_cx, final_cy), saliency_ratio, saliency_map, rect, sal_top, sal_bottom

    # =======================================================
    # 策略函数 1: 运镜模式 (V56 - Zoom + Pan + Ken Burns)
    # =======================================================
    def _apply_zoom_pan_strategy(self, clip, analysis_data):
        """
        封装 V56 的逻辑：适用于风景、静态人物、需要氛围感的镜头
        """
        print("   [Strategy] 激活模式: Zoom & Pan (Ken Burns)")
        src_w, src_h = clip.w, clip.h
        output_aspect = self.target_w / self.target_h
        
        # 解包基础数据
        smooth_cx = analysis_data['smooth_cx']
        smooth_ratios = analysis_data['smooth_ratios']
        raw_tops = analysis_data['raw_tops']
        smooth_bottoms = analysis_data['smooth_bottoms']
        smooth_lift_intensity = analysis_data['smooth_lift_intensity']
        raw_rects = analysis_data['raw_rects']
        smooth_tops_ref = self.smooth_data(raw_tops, int(clip.fps * 3.0))

        total_frames = len(smooth_cx)
        
        # 定义时序参数
        ZOOM_DURATION = min(clip.duration * 0.4, 2.5) 
        PAN_START_TIME = ZOOM_DURATION * 0.8
        PAN_END_TIME = clip.duration - 0.2
        
        # 记录日志: 变焦事件
        avg_ratio = np.mean(smooth_ratios)
        is_landscape_mode = avg_ratio < 0.10
        
        # 记录日志: 初始化事件
        self.logger.log_event("STRATEGY_SELECT", 0, clip.duration, {
            "mode": "ZOOM_PAN_V56",
            "is_landscape": is_landscape_mode
        })

        inst_target_h, inst_target_cx, inst_target_cy = [], [], []
        action_trace = []
        last_accepted_cy = src_h / 2
        
        # 记录 Zoom 参数以便 Log
        zoom_target_scale = 1.0

        for i in range(total_frames):
            current_time = i / clip.fps
            intensity = smooth_lift_intensity[i]
            
            # 垂直混合
            curr_top = raw_tops[i] * intensity + smooth_tops_ref[i] * (1 - intensity)
            
            # 人物/物体形态判断
            subject_h = smooth_bottoms[i] - curr_top
            raw_w = raw_rects[i][2] if raw_rects[i][2] > 1 else 1
            subject_aspect = subject_h / raw_w
            height_ratio = subject_h / src_h
            
            is_face_closeup = (height_ratio > 0.5) and (subject_aspect < 1.6)
            is_tall_body = (height_ratio > 0.6) and (subject_aspect >= 1.6)
            if intensity > 0.3: is_face_closeup = False

            # --- 功能 1: Zoom 计算 ---
            # V56 缩放逻辑
            if is_face_closeup:
                target_scale = 1.0
            else:
                target_scale = 0.55 / (smooth_ratios[i] + 0.25)
                duration_limit = 1.0 + (clip.duration * 0.5)
                hard_limit = 2.0 if is_tall_body else 1.7
                final_limit = min(duration_limit, hard_limit)
                target_scale = max(1.15, min(target_scale, final_limit))
            
            zoom_target_scale = target_scale # Update for log

            if current_time < ZOOM_DURATION:
                zoom_raw = current_time / ZOOM_DURATION
                real_scale_factor = self.ease_out_cubic(zoom_raw)
                action_trace.append("ZOOM-IN")
            else:
                real_scale_factor = 1.0
                action_trace.append("PAN/HOLD")
            
            current_scale = 1.0 + (target_scale - 1.0) * real_scale_factor
            t_h = src_h / current_scale
            t_h = max(t_h, 480)

            # --- 功能 2: Vertical Anti-Bounce (垂直防抖) ---
            anchor_bias = self.map_range(current_scale, 1.0, 1.5, 0.45, 0.25)
            if intensity > 0.1: anchor_bias = self.map_range(intensity, 0.0, 1.0, anchor_bias, 0.20)
            
            proposed_t_cy = curr_top + (subject_h * anchor_bias)
            
            # 头部/脚部保护 (Head/Foot Guards)
            feet_buffer_ratio = 0.18
            safe_chin_bottom = smooth_bottoms[i] + (t_h * feet_buffer_ratio)
            headroom_ratio = 0.05 if is_face_closeup else (0.35 if intensity > 0.1 else 0.12)
            safe_head_top = curr_top - (t_h * headroom_ratio)
            
            # 冲突处理
            temp_top = proposed_t_cy - t_h/2
            if temp_top > safe_head_top: proposed_t_cy -= (temp_top - safe_head_top)
            
            temp_bottom = proposed_t_cy + t_h/2
            if temp_bottom < safe_chin_bottom:
                needed_shift = safe_chin_bottom - temp_bottom
                max_allowed_shift = (proposed_t_cy - t_h/2) - safe_head_top + (t_h * 0.05)
                proposed_t_cy += min(needed_shift, max_allowed_shift)

            # Deadband 逻辑
            bounce_threshold = src_h * 0.02
            if i == 0:
                t_cy = proposed_t_cy
                last_accepted_cy = t_cy
            else:
                diff = abs(proposed_t_cy - last_accepted_cy)
                if diff < bounce_threshold and intensity < 0.1:
                    t_cy = last_accepted_cy
                else:
                    t_cy = last_accepted_cy + (proposed_t_cy - last_accepted_cy) * 0.1
                    last_accepted_cy = t_cy
            
            # 物理边界
            if (t_cy - t_h/2) < 0: t_cy = t_h/2
            if (t_cy + t_h/2) > src_h: t_cy = src_h - t_h/2

            # --- 功能 3: Panning (水平运镜) ---
            if is_landscape_mode:
                subject_x = smooth_cx[i]
                center_x = src_w / 2
                if current_time < PAN_START_TIME:
                    pan_t = 0.0
                else:
                    pan_duration = PAN_END_TIME - PAN_START_TIME
                    pt = (current_time - PAN_START_TIME) / (pan_duration if pan_duration>0 else 1)
                    pan_t = self.ease_sine_in_out(max(0.0, min(1.0, pt)))
                
                t_cx = center_x * (1 - pan_t) + subject_x * pan_t
                
                # Landscape 边缘处理
                view_w = t_h * output_aspect
                l, r = t_cx - view_w/2, t_cx + view_w/2
                sl, sr = subject_x - subject_h*0.25, subject_x + subject_h*0.25
                if sl < l: t_cx -= (l - sl)
                elif sr > r: t_cx += (sr - r)
            else:
                t_cx = smooth_cx[i]
                # Center Magnet
                if abs(t_cx - src_w/2) < (src_w * 0.05): t_cx = src_w/2
            
            view_w = t_h * output_aspect
            if (t_cx - view_w/2) < 0: t_cx = view_w/2
            if (t_cx + view_w/2) > src_w: t_cx = src_w - view_w/2

            inst_target_h.append(t_h)
            inst_target_cx.append(t_cx)
            inst_target_cy.append(t_cy)

        # 记录关键事件
        self.logger.log_event("ZOOM_PHASE", 0, ZOOM_DURATION, {"max_scale": zoom_target_scale, "curve": "ease_out_cubic"})
        if is_landscape_mode:
            self.logger.log_event("PAN_PHASE", PAN_START_TIME, PAN_END_TIME, {"target": "Subject_X", "curve": "ease_sine_in_out"})

        return inst_target_h, inst_target_cx, inst_target_cy, action_trace

    # =======================================================
    # 策略函数 2: 跟随模式 (V32 - Active Tracking)
    # =======================================================
    def _apply_tracking_strategy(self, clip, analysis_data):
        """
        封装 V32 的逻辑：适用于人物大幅移动、舞蹈、Vlog 演讲
        """
        print("   [Strategy] 激活模式: Active Tracking (V32)")
        src_w, src_h = clip.w, clip.h
        output_aspect = self.target_w / self.target_h
        
        smooth_cx = analysis_data['smooth_cx']
        smooth_cy = analysis_data['smooth_cy']
        smooth_ratios = analysis_data['smooth_ratios']
        smooth_tops = analysis_data['smooth_tops']
        smooth_bottoms = analysis_data['smooth_bottoms']
        
        # 记录日志
        self.logger.log_event("STRATEGY_SELECT", 0, clip.duration, {"mode": "TRACKING_V32", "note": "High motion detected"})

        inst_target_h, inst_target_cx, inst_target_cy = [], [], []
        action_trace = []

        for i in range(len(smooth_cx)):
            curr_ratio = smooth_ratios[i]
            
            # [V32] Natural Scale Calculation
            target_scale = 0.5 / (curr_ratio + 0.25)
            duration_limit = 1.0 + (clip.duration * 0.08)
            hard_limit = 1.6
            target_scale = max(1.1, min(target_scale, min(duration_limit, hard_limit)))
            
            target_h = src_h / target_scale
            target_h = max(target_h, 480)
            target_w = target_h * output_aspect
            
            # [V32] Portrait Focus Logic (Vertical)
            curr_top = smooth_tops[i]
            curr_bottom = smooth_bottoms[i]
            full_height = curr_bottom - curr_top
            
            essential_bottom = curr_top + (full_height * 0.75)
            required_min_h = (essential_bottom - curr_top) * 1.2
            if target_h < required_min_h: target_h = required_min_h
            if target_h > src_h: target_h = src_h
            
            bias_y = target_h * -0.05
            target_cy = smooth_cy[i] + bias_y
            
            # Head Guard (Priority)
            head_buffer = target_h * 0.12
            safe_head_top = curr_top - head_buffer
            if (target_cy - target_h/2) > safe_head_top:
                target_cy -= ((target_cy - target_h/2) - safe_head_top)
                
            # Screen Clamp
            if (target_cy - target_h/2) < 0: target_cy = target_h/2
            if (target_cy + target_h/2) > src_h: target_cy = src_h - target_h/2
            
            target_cx = smooth_cx[i]
            # Screen Clamp X
            if (target_cx - target_w/2) < 0: target_cx = target_w/2
            if (target_cx + target_w/2) > src_w: target_cx = src_w - target_w/2
            
            inst_target_h.append(target_h)
            inst_target_cx.append(target_cx)
            inst_target_cy.append(target_cy)
            
            # 插值计算 (V32 Style)
            linear_prog = i / len(smooth_cx)
            eased_prog = self.ease_quintic_in_out(linear_prog)
            
            # 这里我们只存目标值，后续统一做平滑
            action_trace.append("TRACKING")
            
        return inst_target_h, inst_target_cx, inst_target_cy, action_trace

    def analyze_content(self, clip, force_mode=None):
        print(f"   [Phase 1] 扫描全片并分析最佳策略...")
        src_w, src_h = clip.w, clip.h
        self.logger.set_file(clip.filename)
        
        # 1. 采集原始数据
        raw_cx, raw_cy, raw_ratios, raw_rects = [], [], [], []
        raw_tops, raw_bottoms = [], []
        
        step = 2
        for i, frame in enumerate(clip.iter_frames()):
            if i % step == 0:
                small_frame = cv2.resize(frame, (640, 360))
                center, ratio, _, v_rect, s_top, s_bot = self.get_saliency_roi(small_frame)
                scale_f = src_h / 360.0
                
                cx = center[0] * (src_w / 640.0)
                cy = center[1] * scale_f
                rx, ry, rw, rh = v_rect
                rect = [rx*(src_w/640.0), ry*scale_f, rw*(src_w/640.0), rh*scale_f]
                
                raw_cx.append(cx); raw_cy.append(cy)
                raw_ratios.append(ratio); raw_rects.append(rect)
                raw_tops.append(s_top * scale_f); raw_bottoms.append(s_bot * scale_f)
            else:
                # Fill gaps
                raw_cx.append(raw_cx[-1] if raw_cx else src_w/2)
                raw_cy.append(raw_cy[-1] if raw_cy else src_h/2)
                raw_ratios.append(raw_ratios[-1] if raw_ratios else 0.5)
                raw_rects.append(raw_rects[-1] if raw_rects else [0,0,0,0])
                raw_tops.append(raw_tops[-1] if raw_tops else 0)
                raw_bottoms.append(raw_bottoms[-1] if raw_bottoms else src_h)

        # 2. 预处理数据 (Smoothing)
        win_pos = int(clip.fps * 4.0)
        win_ratio = int(clip.fps * 4.0)
        
        smooth_cx = self.smooth_data(raw_cx, win_pos)
        smooth_cy = self.smooth_data(raw_cy, int(clip.fps * 3.0))
        smooth_ratios = self.smooth_data(raw_ratios, win_ratio)
        smooth_tops = self.smooth_data(raw_tops, int(clip.fps * 3.0))
        smooth_bottoms = self.smooth_data(raw_bottoms, win_pos)
        
        # 垂直速度分析 (for Bounce detection)
        d_top = np.diff(np.array(raw_tops), prepend=raw_tops[0])
        smooth_d_top = self.smooth_data(d_top, int(clip.fps * 0.5), 1)
        lift_intensity = []
        for v in smooth_d_top:
            if v > -1.0: val = 0.0
            elif v < -4.0: val = 1.0
            else: val = (abs(v) - 1.0) / 3.0
            lift_intensity.append(val)
        smooth_lift = self.smooth_data(lift_intensity, int(clip.fps * 2.5), 1)

        # 3. 策略选择 (Decision Making)
        dx = np.diff(raw_cx); dy = np.diff(raw_cy)
        motion_score = np.mean(np.sqrt(dx**2 + dy**2))
        
        analysis_packet = {
            "smooth_cx": smooth_cx, "smooth_cy": smooth_cy,
            "smooth_ratios": smooth_ratios,
            "raw_rects": raw_rects, "raw_tops": raw_tops,
            "smooth_tops": smooth_tops, "smooth_bottoms": smooth_bottoms,
            "smooth_lift_intensity": smooth_lift
        }
        
        # 选择逻辑
        if force_mode == "TRACK":
            t_h, t_cx, t_cy, trace = self._apply_tracking_strategy(clip, analysis_packet)
        elif force_mode == "ZOOM_PAN":
            t_h, t_cx, t_cy, trace = self._apply_zoom_pan_strategy(clip, analysis_packet)
        else:
            # 自动判断
            if motion_score > 12.0: # 高运动 -> Tracking (V32)
                t_h, t_cx, t_cy, trace = self._apply_tracking_strategy(clip, analysis_packet)
                mode_name = "TRACKING"
            else: # 低运动 -> Zoom/Pan (V56)
                t_h, t_cx, t_cy, trace = self._apply_zoom_pan_strategy(clip, analysis_packet)
                mode_name = "ZOOM_PAN"

        # 4. 后处理平滑 (Final Smoothing)
        # V32 使用 ease 插值，V56 使用 path 计算，这里统一加一层 SavGol 保证输出如丝般顺滑
        win_final = int(clip.fps * 0.5)
        final_h = self.smooth_data(t_h, win_final, 2)
        final_cx = self.smooth_data(t_cx, win_final, 2)
        final_cy = self.smooth_data(t_cy, win_final, 2)
        
        # 5. 生成 Crop Box
        crop_rects = []
        final_targets = []
        output_aspect = self.target_w / self.target_h
        
        for i in range(len(final_h)):
            fh, fcx, fcy = final_h[i], final_cx[i], final_cy[i]
            fw = fh * output_aspect
            x1 = fcx - fw/2
            y1 = fcy - fh/2
            crop_rects.append([x1, y1, fw, fh])
            final_targets.append((fcx, fcy, fw, fh))
            
        crop_rects = np.array(crop_rects)
        
        # 保存日志
        log_name = os.path.splitext(os.path.basename(clip.filename))[0] + "_log.json"
        self.logger.save_to_disk(log_name)

        return {
            "crop_box": (crop_rects[:,0], crop_rects[:,1], crop_rects[:,2], crop_rects[:,3]),
            "target_box": final_targets,
            "raw_rects": raw_rects,
            "raw_ratios": smooth_ratios,
            "raw_tops": smooth_tops,
            "raw_bottoms": smooth_bottoms,
            "action_trace": trace,
            "mode": mode_name if not force_mode else force_mode
        }

    def render_hud_monitor(self, clip, data):
        """ 渲染 HUD (融合 V32/V56 风格) """
        cx, cy, cw, ch = data["crop_box"]
        targets = data["target_box"]
        raw_rects = data["raw_rects"]
        ratios = data["raw_ratios"]
        smooth_tops = data["raw_tops"]
        smooth_bottoms = data["raw_bottoms"]
        mode = data["mode"]
        action_trace = data["action_trace"]
        
        src_w, src_h = clip.w, clip.h
        
        def frame_process(get_frame, t):
            raw_frame = get_frame(t)
            idx = min(int(t * clip.fps), len(cx)-1)
            
            # 热力图生成
            small_viz = cv2.resize(raw_frame, (320, 180))
            _, _, s_map_small, _, _, _ = self.get_saliency_roi(small_viz)
            heatmap = cv2.applyColorMap(s_map_small, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (src_w, src_h))
            monitor_bg = cv2.addWeighted(raw_frame, 0.60, heatmap, 0.40, 0)
            
            # 坐标变换
            ix1, iy1 = int(cx[idx]), int(cy[idx])
            ix2, iy2 = int(cx[idx]+cw[idx]), int(cy[idx]+ch[idx])
            rx1, ry1 = max(0, ix1), max(0, iy1)
            rx2, ry2 = min(src_w, ix2), min(src_h, iy2)
            if rx2 <= rx1 or ry2 <= ry1: rx1,ry1,rx2,ry2 = 0,0,src_w,src_h
            
            try:
                crop_img = raw_frame[ry1:ry2, rx1:rx2]
                final_res_img = cv2.resize(crop_img, (self.target_w, self.target_h))
            except:
                final_res_img = cv2.resize(raw_frame, (self.target_w, self.target_h))

            # Monitor View
            scale = self.target_h / src_h
            mon_w = int(src_w * scale)
            mon_img = cv2.resize(monitor_bg, (mon_w, self.target_h))
            def to_mon(x, y): return int(x * scale), int(y * scale)
            
            # Draw Lines
            if idx < len(smooth_tops):
                _, m_head_y = to_mon(0, smooth_tops[idx])
                cv2.line(mon_img, (0, m_head_y), (mon_w, m_head_y), (255, 0, 255), 1)
            _, m_feet_y = to_mon(0, smooth_bottoms[idx])
            cv2.line(mon_img, (0, m_feet_y), (mon_w, m_feet_y), (255, 0, 0), 1)
            
            # Target Box (Yellow Dashed)
            tcx, tcy, tcw, tch = targets[idx]
            tx1, ty1 = int(tcx - tcw/2), int(tcy - tch/2)
            tx2, ty2 = int(tcx + tcw/2), int(tcy + tch/2)
            mtx1, mty1 = to_mon(tx1, ty1); mtx2, mty2 = to_mon(tx2, ty2)
            draw_dashed_rect(mon_img, mtx1, mty1, mtx2, mty2, (0, 255, 255), 1)
            
            # Current Crop Box (Green Solid)
            mx1, my1 = to_mon(ix1, iy1); mx2, my2 = to_mon(ix2, iy2)
            cv2.rectangle(mon_img, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            
            # Vector Arrow
            curr_center = ((mx1+mx2)//2, (my1+my2)//2)
            target_center = ((mtx1+mtx2)//2, (mty1+mty2)//2)
            status = action_trace[idx]
            arrow_c = (0,0,255) if "ZOOM" in status or "PAN" in status else (0,255,255)
            draw_vector_arrow(mon_img, curr_center, target_center, arrow_c)

            # UI HUD
            ui_x, ui_y = 20, self.target_h - 60
            curr_zoom = src_h / ch[idx]
            draw_hud_label(mon_img, f"ZOOM: {curr_zoom:.2f}x", ui_x, ui_y, bg_color=(0,100,0))
            draw_hud_label(mon_img, f"MODE: {mode}", ui_x + 130, ui_y, bg_color=(100, 0, 100))
            draw_hud_label(mon_img, f"ACT: {status}", ui_x + 280, ui_y, bg_color=(0, 0, 100))
            
            draw_progress_bar(mon_img, ui_x, ui_y + 20, 150, 10, ratios[idx], 0.6, (0, 200, 255), "SALIENCY")
            
            combined = np.zeros((self.target_h, self.target_w * 2, 3), dtype=np.uint8)
            off_x = (self.target_w - mon_w) // 2
            combined[:, off_x:off_x+mon_w] = mon_img
            combined[:, self.target_w:] = final_res_img
            return combined

        return clip.fl(frame_process)

if __name__ == "__main__":
    engine = DirectorEngine()
    
    # 请填入视频文件路径
    VIDEO_PLAYLIST = [
        r"C:\Users\Administrator\Desktop\vlog-clean\data\5.mp4", # 测试 Tracking
        r"C:\Users\Administrator\Desktop\vlog-clean\data\7.mp4", # 测试 Zoom/Pan
    ]
    
    print("\n" + "="*60)
    print("【AI 导演引擎 V60.2 - Merged Ultimate (NumPy 2.0 Fixed)】")
    print("  包含: V32 Tracking + V56 Ken Burns + Robust Logging")
    print("="*60 + "\n")
    
    for i, video_path in enumerate(VIDEO_PLAYLIST):
        print(f">>> [{i+1}/{len(VIDEO_PLAYLIST)}] 处理: {os.path.basename(video_path)}")
        if not os.path.exists(video_path): continue
            
        try:
            clip = VideoFileClip(video_path)
            
            # 这里的 analyze_content 会自动判断模式并记录 JSON
            data = engine.analyze_content(clip) 
            
            hud_clip = engine.render_hud_monitor(clip, data)
            
            out_dir = "Merged_Output_V60"
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            fname = os.path.splitext(os.path.basename(video_path))[0]
            out_name = os.path.join(out_dir, f"V60_{fname}_{data['mode']}.mp4")
            
            hud_clip.write_videofile(out_name, codec='libx264', audio_codec='aac', fps=24, threads=4)
            clip.close()
            hud_clip.close()
            print("√ 完成\n")
            
        except Exception as e:
            print(f"!!! Error: {e}")
            import traceback
            traceback.print_exc()