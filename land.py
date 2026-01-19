import cv2
import numpy as np
import os
import sys
import time
import argparse
import onnxruntime as ort
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter
from collections import deque

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
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)
    cx, cy = (x1+x2)//2, (y1+y2)//2
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 15, 1)

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
# 核心算法引擎 V67.0 (Fluid Motion)
# ==========================================

class DirectorEngine:
    def __init__(self, target_res=(1920, 1080), forced_mode='auto'):
        print(f"[System] 初始化导演引擎 V67.0 (Fluid Motion)...")
        print(f"[System] 运镜模式设定: {forced_mode.upper()}")
        self.target_w, self.target_h = target_res
        self.forced_mode = forced_mode.lower() 
        try:
            self.saliency_detector = U2NetSaliency('u2netp.onnx')
            print("   [Model] U2NetP Active.")
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            sys.exit(1)

    def smooth_data(self, data, window_size, polyorder=3):
        data_len = len(data)
        if data_len < 4: return data 
        if window_size >= data_len: window_size = data_len - 1
        if window_size % 2 == 0: window_size -= 1
        if window_size < 3: return data
        if polyorder >= window_size: polyorder = window_size - 1
        return savgol_filter(data, window_size, polyorder)

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

    def calculate_morphology_scale(self, ratio, duration, is_face_closeup, is_tall_body, is_landscape=False):
        if is_face_closeup: return 1.0
        target_scale = 0.55 / (ratio + 0.25)
        speed_limit = 0.5 
        duration_limit = 1.0 + (duration * speed_limit) 
        hard_limit = 1.7
        if is_tall_body: hard_limit = 2.0 
        final_limit = min(duration_limit, hard_limit)
        target_scale = max(1.15, min(target_scale, final_limit))
        return target_scale

    def ease_out_cubic(self, t):
        return 1 - pow(1 - t, 3)

    def ease_sine_in_out(self, t):
        return -(np.cos(np.pi * t) - 1) / 2

    def map_range(self, val, in_min, in_max, out_min, out_max):
        if val <= in_min: return out_min
        if val >= in_max: return out_max
        return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def analyze_content(self, clip):
        print(f"   [Phase 1] 扫描全片 (V67.0 Fluid Motion)...")
        src_w, src_h = clip.w, clip.h
        
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
                real_top = s_top * scale_f
                real_bot = s_bot * scale_f
                raw_cx.append(cx); raw_cy.append(cy)
                raw_ratios.append(ratio); raw_rects.append(rect)
                raw_tops.append(real_top); raw_bottoms.append(real_bot)
            else:
                raw_cx.append(raw_cx[-1] if raw_cx else src_w/2)
                raw_cy.append(raw_cy[-1] if raw_cy else src_h/2)
                raw_ratios.append(raw_ratios[-1] if raw_ratios else 0.5)
                raw_rects.append(raw_rects[-1] if raw_rects else [0,0,0,0])
                raw_tops.append(raw_tops[-1] if raw_tops else 0)
                raw_bottoms.append(raw_bottoms[-1] if raw_bottoms else src_h)

        # =========================================================================
        # [V65/V67] 结构完整性检查 + 动量清洗
        # =========================================================================
        print("   [Phase 1.5] 执行结构完整性锁定与动量修复...")
        
        clean_tops, clean_bottoms = [], []
        clean_cx, clean_cy = [], []
        
        MAX_JUMP_Y = src_h * 0.15 
        MIN_CONFIDENCE = 0.002
        
        last_valid_top = raw_tops[0]
        last_valid_bottom = raw_bottoms[0]
        last_valid_cx = raw_cx[0]
        last_valid_cy = raw_cy[0]
        
        height_history = deque(maxlen=10)
        init_h = max(1.0, last_valid_bottom - last_valid_top)
        for _ in range(10): height_history.append(init_h)
        
        vx_history = [] 
        MAX_HISTORY = 8
        
        coasting_state_counter = 0
        STABILITY_THRESHOLD = 5 
        is_coasting_locked = False 
        
        for k in range(len(raw_tops)):
            curr_top = raw_tops[k]
            curr_bot = raw_bottoms[k]
            curr_r = raw_ratios[k]
            curr_cx = raw_cx[k]
            curr_height = curr_bot - curr_top
            if curr_height < 1: curr_height = 1
            
            # 异常判定
            is_low_conf = curr_r < MIN_CONFIDENCE
            jump_too_big = abs(curr_top - last_valid_top) > MAX_JUMP_Y
            hit_floor_suddenly = (last_valid_bottom < src_h * 0.85) and (curr_bot > src_h * 0.98)
            
            avg_h = np.mean(height_history)
            is_box_collapsed = curr_height < (avg_h * 0.65)
            top_dropped_badly = (curr_top > last_valid_top + (avg_h * 0.15)) and is_box_collapsed
            
            current_frame_is_bad = is_low_conf or jump_too_big or hit_floor_suddenly or is_box_collapsed or top_dropped_badly
            
            # 状态机去抖
            if current_frame_is_bad: coasting_state_counter += 1
            else: coasting_state_counter -= 1
            coasting_state_counter = max(0, min(coasting_state_counter, STABILITY_THRESHOLD + 2))
            
            if coasting_state_counter >= STABILITY_THRESHOLD: is_coasting_locked = True
            elif coasting_state_counter == 0: is_coasting_locked = False
            
            # 动量计算
            if not current_frame_is_bad:
                inst_vx = curr_cx - last_valid_cx
                vx_history.append(inst_vx)
                if len(vx_history) > MAX_HISTORY: vx_history.pop(0)
            
            # 数据输出
            if is_coasting_locked:
                avg_vx = 0
                if len(vx_history) > 0: avg_vx = np.mean(vx_history)
                pred_cx = last_valid_cx + avg_vx
                pred_cx = max(0, min(pred_cx, src_w))
                
                clean_tops.append(last_valid_top)
                clean_bottoms.append(last_valid_bottom)
                clean_cx.append(pred_cx) 
                clean_cy.append(last_valid_cy)
                last_valid_cx = pred_cx 
            else:
                clean_tops.append(curr_top)
                clean_bottoms.append(curr_bot)
                clean_cx.append(curr_cx)
                clean_cy.append(raw_cy[k])
                
                last_valid_top = curr_top
                last_valid_bottom = curr_bot
                last_valid_cx = curr_cx
                last_valid_cy = raw_cy[k]
                height_history.append(curr_height)

        # =========================================================================
        # 平滑处理 & 模式覆盖
        # =========================================================================
        
        clean_tops_arr = np.array(clean_tops)
        d_top = np.diff(clean_tops_arr, prepend=clean_tops_arr[0])
        smooth_d_top = self.smooth_data(d_top, int(clip.fps * 0.5), 1)
        
        motion_score = np.mean(np.abs(d_top))
        
        auto_mode = "DOLLY"
        if motion_score > 15.0: auto_mode = "LOCKED"
        
        if self.forced_mode == 'locked': final_mode = "LOCKED"
        elif self.forced_mode == 'dolly': final_mode = "DOLLY"
        elif self.forced_mode == 'fixed': final_mode = "FIXED"
        else: final_mode = auto_mode 

        need_slowmo = motion_score > 8.0 and clip.duration > 4

        # 平滑参数
        if final_mode == "LOCKED":
            win_pos = int(clip.fps * 1.5) 
            win_cx_smooth = int(clip.fps * 1.5)
        elif final_mode == "FIXED":
            win_pos = int(clip.fps * 6.0)
            win_cx_smooth = int(clip.fps * 6.0)
        else:
            win_pos = int(clip.fps * 4.0)
            win_cx_smooth = int(clip.fps * 4.0)

        smooth_cx = self.smooth_data(clean_cx, win_cx_smooth) 
        smooth_cy_ref = self.smooth_data(clean_cy, int(clip.fps * 3.0)) 
        smooth_ratios = self.smooth_data(raw_ratios, int(clip.fps * 4.0)) 
        smooth_tops_ref = self.smooth_data(clean_tops, int(clip.fps * 3.0)) 
        smooth_bottoms = self.smooth_data(clean_bottoms, win_pos)
        
        lift_intensity_list = []
        for v in smooth_d_top:
            if v > -1.0: val = 0.0
            elif v < -4.0: val = 1.0
            else: val = (abs(v) - 1.0) / 3.0
            lift_intensity_list.append(val)
        smooth_lift_intensity = self.smooth_data(lift_intensity_list, int(clip.fps * 2.5), 1)
        
        inst_target_h = []
        inst_target_cy = []
        inst_target_cx = []
        active_tops_trace = [] 
        action_trace = []
        
        output_aspect = self.target_w / self.target_h
        total_frames = len(smooth_cx)
        
        # [V67] 优化时序：加长 Zoom，重叠 Pan
        # 原则：Zoom 变慢，Pan 提前介入，消除中间停顿
        ZOOM_DURATION = min(clip.duration * 0.55, 3.5) # 加长 Zoom 时间
        PAN_START_TIME = ZOOM_DURATION * 0.85          # 提前 15% 开始平移
        
        dolly_locked_cy = None 
        
        # [V66/V67] 构图偏置计算
        start_on_left = smooth_cx[0] < (src_w / 2)
        OFFSET_RATIO = 0.166 
        
        for i in range(total_frames):
            curr_ratio = smooth_ratios[i]
            current_time = i / clip.fps
            intensity = smooth_lift_intensity[i]
            
            curr_top = clean_tops[i] * intensity + smooth_tops_ref[i] * (1 - intensity)
            curr_bottom = smooth_bottoms[i]
            subject_h = curr_bottom - curr_top
            
            is_coasting = (raw_ratios[i] < MIN_CONFIDENCE)
            is_landscape_mode = curr_ratio < 0.10
            
            status_tag = "STANDARD"
            
            # =========================================
            # [V67] 智能 DOLLY 模式 - 流畅无停顿
            # =========================================
            if final_mode == "DOLLY":
                status_tag = "DOLLY-FLUID"
                
                # 1. 最小包围盒
                raw_w = raw_rects[i][2] if raw_rects[i][2] > 1 else 1
                min_h_vertical = subject_h * 1.10
                min_h_horizontal = (raw_w * 1.10) / output_aspect
                safe_min_h = max(min_h_vertical, min_h_horizontal)
                
                # 2. 变焦计算
                safe_scale = self.calculate_morphology_scale(curr_ratio, clip.duration, False, False, is_landscape=is_landscape_mode)
                
                if current_time < ZOOM_DURATION:
                    zoom_raw = current_time / ZOOM_DURATION
                    real_scale_factor = self.ease_out_cubic(zoom_raw)
                else:
                    real_scale_factor = 1.0
                
                current_scale = 1.0 + (safe_scale - 1.0) * real_scale_factor
                t_h_calc = src_h / current_scale
                
                t_h = max(t_h_calc, safe_min_h)
                t_h = min(t_h, src_h) 
                
                current_view_w = t_h * output_aspect
                
                # 3. 垂直锁定 (Y-Axis Lock)
                # 逻辑：完全不变，变焦结束后锁死
                subject_mid_y = curr_top + (subject_h / 2)
                vertical_velocity = abs(smooth_d_top[i])
                VIOLENT_THRESHOLD = 5.0
                is_violent = vertical_velocity > VIOLENT_THRESHOLD
                
                if current_time < ZOOM_DURATION:
                    t_cy = subject_mid_y
                    dolly_locked_cy = t_cy 
                else:
                    if is_violent:
                        t_cy = subject_mid_y
                        dolly_locked_cy = t_cy 
                        status_tag = "DOLLY-UNLOCK"
                    else:
                        t_cy = dolly_locked_cy
                
                # 4. 水平逻辑 (Overlap Panning)
                # Zoom 阶段偏置 -> Overlap 阶段平滑归零
                
                offset_val = current_view_w * OFFSET_RATIO
                if start_on_left: target_offset = offset_val 
                else: target_offset = -offset_val 
                
                if current_time < PAN_START_TIME:
                    # 纯变焦期：保持 1/3 偏置
                    current_offset = target_offset
                else:
                    # 混合期：开始向中心平移
                    # 确保平移时长足够，直到视频结束或至少 3 秒
                    pan_dur = max(2.0, clip.duration - PAN_START_TIME)
                    pan_prog = (current_time - PAN_START_TIME) / pan_dur
                    pan_prog = max(0.0, min(1.0, pan_prog))
                    ease_pan = self.ease_sine_in_out(pan_prog)
                    
                    current_offset = target_offset * (1.0 - ease_pan)
                    status_tag = "PANNING-BLEND"
                
                t_cx = smooth_cx[i] + current_offset
                
            elif final_mode == "FIXED":
                status_tag = "FIXED-CAM"
                t_cx = src_w / 2
                t_cy = src_h / 2
                t_h = src_h / 1.1
                
            else:
                # -------------------------------------
                # LOCKED / STANDARD (Legacy)
                # -------------------------------------
                if is_coasting: status_tag = "MOMENTUM"
                elif is_landscape_mode: status_tag = "PANNING"
                
                height_ratio = subject_h / src_h
                raw_w = raw_rects[i][2] if raw_rects[i][2] > 1 else 1
                subject_aspect = subject_h / raw_w
                is_face_closeup = (height_ratio > 0.5) and (subject_aspect < 1.6)
                is_tall_body = (height_ratio > 0.6) and (subject_aspect >= 1.6)
                if intensity > 0.3: is_face_closeup = False
                
                safe_scale = self.calculate_morphology_scale(curr_ratio, clip.duration, is_face_closeup, is_tall_body, is_landscape=is_landscape_mode)
                
                if current_time < ZOOM_DURATION:
                    zoom_raw = current_time / ZOOM_DURATION
                    real_scale_factor = self.ease_out_cubic(zoom_raw)
                else:
                    real_scale_factor = 1.0
                
                current_scale = 1.0 + (safe_scale - 1.0) * real_scale_factor
                t_h = src_h / current_scale
                t_h = max(t_h, 480)
                
                anchor_bias = self.map_range(current_scale, 1.0, 1.5, 0.45, 0.25)
                if intensity > 0.1: anchor_bias = self.map_range(intensity, 0.0, 1.0, anchor_bias, 0.20)
                proposed_t_cy = curr_top + (subject_h * anchor_bias)
                
                feet_buffer_ratio = 0.18 
                safe_chin_bottom = smooth_bottoms[i] + (t_h * feet_buffer_ratio)
                if intensity > 0.1: headroom_ratio = 0.35 
                elif is_face_closeup: headroom_ratio = 0.05
                else: headroom_ratio = 0.12
                safe_head_top = curr_top - (t_h * headroom_ratio)
                
                temp_top = proposed_t_cy - t_h/2
                if temp_top > safe_head_top: proposed_t_cy -= (temp_top - safe_head_top)
                temp_bottom = proposed_t_cy + t_h/2
                if temp_bottom < safe_chin_bottom:
                    needed = safe_chin_bottom - temp_bottom
                    allowed = (proposed_t_cy - t_h/2) - safe_head_top + (t_h * 0.05)
                    if needed > 0: proposed_t_cy += min(needed, allowed)
                
                bounce_threshold = src_h * 0.02 
                if i == 0: t_cy = proposed_t_cy; last_accepted_cy = t_cy
                else:
                    diff = abs(proposed_t_cy - last_accepted_cy)
                    if diff < bounce_threshold and intensity < 0.1: t_cy = last_accepted_cy 
                    else:
                        alpha = 0.1 
                        t_cy = last_accepted_cy + (proposed_t_cy - last_accepted_cy) * alpha
                        last_accepted_cy = t_cy
                
                if is_coasting: t_cx = smooth_cx[i] 
                elif is_landscape_mode:
                    center_x = src_w / 2
                    subject_x = smooth_cx[i]
                    if current_time < PAN_START_TIME: pan_t = 0.0 
                    else:
                        dur = PAN_END_TIME - PAN_START_TIME
                        pan_t = 1.0 if dur <= 0 else self.ease_sine_in_out(max(0.0, min(1.0, (current_time - PAN_START_TIME)/dur)))
                    t_cx = center_x * (1 - pan_t) + subject_x * pan_t
                else:
                    t_cx = smooth_cx[i]
                    if abs(t_cx - src_w/2) < (src_w * 0.05): t_cx = src_w/2
            
            # Global Boundary Protection
            if (t_cy - t_h/2) < 0: t_cy = t_h/2
            if (t_cy + t_h/2) > src_h: t_cy = src_h - t_h/2
            view_w = t_h * output_aspect
            if (t_cx - view_w/2) < 0: t_cx = view_w/2
            if (t_cx + view_w/2) > src_w: t_cx = src_w - view_w/2
            
            inst_target_h.append(t_h)
            inst_target_cy.append(t_cy)
            inst_target_cx.append(t_cx)
            active_tops_trace.append(curr_top)
            action_trace.append(status_tag)

        # Output Smoothing
        if final_mode == "DOLLY":
            smooth_win = int(clip.fps * 2.0)
        else:
            smooth_win = int(clip.fps * 0.5)
            
        smooth_final_h = self.smooth_data(inst_target_h, smooth_win, 2)
        smooth_final_cy = self.smooth_data(inst_target_cy, smooth_win, 2)
        smooth_final_cx = self.smooth_data(inst_target_cx, smooth_win, 2)
        
        crop_rects = []
        final_targets = []
        
        for i in range(len(smooth_cx)):
            final_h = smooth_final_h[i]
            final_w = final_h * output_aspect
            final_cx = smooth_final_cx[i]
            final_cy = smooth_final_cy[i]
            
            cx1 = final_cx - final_w / 2
            cy1 = final_cy - final_h / 2
            
            th_raw = inst_target_h[i]
            tw_raw = th_raw * output_aspect
            tcx_raw = inst_target_cx[i]
            tcy_raw = inst_target_cy[i]
            final_targets.append((tcx_raw, tcy_raw, tw_raw, th_raw))
            
            crop_rects.append([cx1, cy1, final_w, final_h])
            
        crop_rects = np.array(crop_rects)
        
        return {
            "crop_box": (crop_rects[:,0], crop_rects[:,1], crop_rects[:,2], crop_rects[:,3]),
            "target_box": final_targets, 
            "motion_score": motion_score,
            "raw_rects": raw_rects,
            "raw_ratios": smooth_ratios,
            "raw_tops": active_tops_trace,
            "raw_bottoms": smooth_bottoms,
            "rec_slowmo": need_slowmo,
            "mode": final_mode, 
            "action_trace": action_trace
        }

    def render_hud_monitor(self, clip, data):
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
            
            small_viz = cv2.resize(raw_frame, (320, 180))
            _, _, s_map_small, _, _, _ = self.get_saliency_roi(small_viz)
            heatmap = cv2.applyColorMap(s_map_small, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (src_w, src_h))
            monitor_bg = cv2.addWeighted(raw_frame, 0.60, heatmap, 0.40, 0)

            ix1, iy1 = int(cx[idx]), int(cy[idx])
            ix2, iy2 = int(cx[idx]+cw[idx]), int(cy[idx]+ch[idx])
            
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
            def to_mon(x, y): return int(x * scale), int(y * scale)
            
            cv2.line(mon_img, (mon_w//3, 0), (mon_w//3, self.target_h), (100,100,100), 1)
            cv2.line(mon_img, (2*mon_w//3, 0), (2*mon_w//3, self.target_h), (100,100,100), 1)
            cv2.line(mon_img, (0, self.target_h//3), (mon_w, self.target_h//3), (100,100,100), 1)
            cv2.line(mon_img, (0, 2*self.target_h//3), (mon_w, 2*self.target_h//3), (100,100,100), 1)
            
            if idx < len(smooth_tops):
                head_y = smooth_tops[idx]; _, m_head_y = to_mon(0, head_y)
                cv2.line(mon_img, (0, m_head_y), (mon_w, m_head_y), (255, 0, 255), 1)
            
            feet_y = smooth_bottoms[idx]; _, m_feet_y = to_mon(0, feet_y)
            cv2.line(mon_img, (0, m_feet_y), (mon_w, m_feet_y), (255, 0, 0), 1)
            
            tcx, tcy, tcw, tch = targets[idx]
            tx1 = int(tcx - tcw / 2)
            ty1 = int(tcy - tch / 2)
            tx2 = int(tcx + tcw / 2)
            ty2 = int(tcy + tch / 2)
            mtx1, mty1 = to_mon(tx1, ty1); mtx2, mty2 = to_mon(tx2, ty2)
            draw_dashed_rect(mon_img, mtx1, mty1, mtx2, mty2, (0, 255, 255), 1)
            
            mx1, my1 = to_mon(ix1, iy1); mx2, my2 = to_mon(ix2, iy2)
            cv2.rectangle(mon_img, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            
            curr_center = ((mx1+mx2)//2, (my1+my2)//2)
            target_center = ((mtx1+mtx2)//2, (mty1+mty2)//2)
            
            current_action = action_trace[idx]
            if "MOMENTUM" in current_action:
                arrow_color = (255, 0, 255) 
                thickness = 4
                tip = 0.5
            elif "DOLLY" in current_action:
                arrow_color = (0, 255, 0) # Green
                thickness = 3
                tip = 0.4
            elif "PAN" in current_action:
                arrow_color = (0, 0, 255) 
                thickness = 5 
                tip = 0.6
            elif "FIXED" in current_action:
                arrow_color = (100, 100, 100)
                thickness = 2
                tip = 0.2
            else:
                arrow_color = (0, 255, 0)
                thickness = 3
                tip = 0.4
            
            draw_vector_arrow(mon_img, curr_center, target_center, arrow_color, thickness, tip)
            
            sx, sy, sw, sh = raw_rects[idx]
            bx1, by1 = to_mon(sx, sy); bx2, by2 = to_mon(sx+sw, sy+sh)
            draw_bracket(mon_img, bx1, by1, bx2, by2, (0, 0, 255), thickness=1)

            ui_x, ui_y = 20, self.target_h - 60
            curr_zoom = src_h / ch[idx]
            draw_hud_label(mon_img, f"ZOOM: {curr_zoom:.2f}x", ui_x, ui_y, bg_color=(0,100,0))
            draw_hud_label(mon_img, f"CAM: {mode}", ui_x + 130, ui_y, bg_color=(100, 0, 100))
            
            status_text = current_action
            bg_stat = (100, 0, 100)
            if "MOMENTUM" in status_text: bg_stat = (200, 0, 200)
            elif "PAN" in status_text: bg_stat = (0, 0, 200)
            elif "DOLLY" in status_text: bg_stat = (0, 100, 0)
            elif "FIXED" in status_text: bg_stat = (50, 50, 50)
            
            draw_hud_label(mon_img, status_text, ui_x + 260, ui_y, bg_color=bg_stat)
            
            draw_progress_bar(mon_img, ui_x, ui_y + 20, 150, 10, ratios[idx], 0.6, (0, 200, 255), "ATTENTION")
            draw_progress_bar(mon_img, ui_x + 170, ui_y + 20, 150, 10, curr_zoom, 2.4, (0, 255, 0), "KB PATH")

            combined = np.zeros((self.target_h, self.target_w * 2, 3), dtype=np.uint8)
            off_x = (self.target_w - mon_w) // 2
            combined[:, off_x:off_x+mon_w] = mon_img
            combined[:, self.target_w:] = final_res_img
            
            return combined

        return clip.fl(frame_process)

# ==========================================
# 主程序
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Director Cam - V67.0 Fluid Motion")
    parser.add_argument('--mode', type=str, default='auto', choices=['auto', 'dolly', 'locked', 'fixed'],
                        help="Choose Camera Movement Mode: auto (default), dolly, locked, fixed")
    
    if 'ipykernel_launcher' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    engine = DirectorEngine(forced_mode=args.mode)
    
    VIDEO_PLAYLIST = [
        r"C:\Users\Administrator\Desktop\vlog-clean\data\5.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\6.mp4"

    ]
    
    print("\n" + "="*60)
    print("【AI 导演监视器 V67.0 - Fluid Motion】")
    print(f"模式: {args.mode.upper()}")
    print(f"待处理视频数: {len(VIDEO_PLAYLIST)}")
    print("="*60 + "\n")
    
    if not os.path.exists("u2netp.onnx"):
        print("!!! 缺少 u2netp.onnx 模型")
        sys.exit(1)
    
    for i, video_path in enumerate(VIDEO_PLAYLIST):
        print(f">>> [{i+1}/{len(VIDEO_PLAYLIST)}] 正在处理: {os.path.basename(video_path)}")
        
        if not os.path.exists(video_path):
            print(f"!!! 警告: 文件不存在，跳过: {video_path}")
            continue
            
        try:
            data = engine.analyze_content(clip := VideoFileClip(video_path))
            hud_clip = engine.render_hud_monitor(clip, data)
            
            out_dir = "hud_v67_output"
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            
            fname = os.path.splitext(os.path.basename(video_path))[0]
            out_name = os.path.join(out_dir, f"HUD_V67_{fname}_{args.mode.upper()}.mp4")
            
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