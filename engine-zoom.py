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
# 模块 1: 工具类与日志系统
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class EventLogger:
    def __init__(self):
        self.logs = {
            "batch_id": time.strftime("%Y%m%d_%H%M%S"),
            "processed_videos": []
        }

    def start_new_video(self, filename, mode):
        self.current_video_log = {
            "file": filename,
            "mode": mode,
            "events": []
        }
        self.logs["processed_videos"].append(self.current_video_log)

    def log_frame(self, frame_idx, time_sec, params):
        if hasattr(self, 'current_video_log'):
            entry = {
                "frame": int(frame_idx),
                "time": float(f"{time_sec:.3f}"),
                "params": params
            }
            self.current_video_log["events"].append(entry)

    def save_to_disk(self, filepath):
        # 保存整个批次的日志
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"   [System] 批次日志已保存至: {filepath}")

# ==========================================
# 模块 2: UI 绘图工具库 (Pro HUD)
# ==========================================

def draw_hud_label(img, text, x, y, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (t_w, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    cv2.rectangle(img, (x, y - t_h - pad), (x + t_w + pad*2, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return x + t_w + pad*2 + 10

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash_len=10):
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        if i % 2 == 0: # Horizontal
            if p1[0] > p2[0]: p1, p2 = p2, p1
            for x in range(int(p1[0]), int(p2[0]), dash_len*2):
                cv2.line(img, (x, int(p1[1])), (min(x+dash_len, int(p2[0])), int(p1[1])), color, thickness)
        else: # Vertical
            if p1[1] > p2[1]: p1, p2 = p2, p1
            for y in range(int(p1[1]), int(p2[1]), dash_len*2):
                cv2.line(img, (int(p1[0]), y), (int(p1[0]), min(y+dash_len, int(p2[1]))), color, thickness)

def draw_vector_arrow(img, p1, p2, color=(0, 255, 255), thickness=4, tip_size=0.3):
    p1 = (int(p1[0]), int(p1[1])); p2 = (int(p2[0]), int(p2[1]))
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if dist > 5.0: 
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
    cv2.drawMarker(img, (int(cx), int(cy)), color, cv2.MARKER_CROSS, 15, 1)

def draw_progress_bar(img, x, y, w, h, val, max_val, color=(0, 255, 0), label=""):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), -1)
    ratio = max(0.0, min(1.0, val / max_val))
    fill_w = int(w * ratio)
    cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 1)
    if label:
        cv2.putText(img, f"{label}: {val:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# ==========================================
# 模块 3: 深度学习显著性检测器 (U2Net)
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
# 模块 4: 核心算法引擎 (CinematicEngine)
# ==========================================

class CinematicEngine:
    def __init__(self, target_res=(1920, 1080)):
        print(f"[System] 初始化 Cinematic Engine (Batch Processing V9)...")
        self.target_w, self.target_h = target_res
        self.output_aspect = self.target_w / self.target_h
        self.logger = EventLogger()
        try:
            self.saliency_detector = U2NetSaliency('u2netp.onnx')
            print("   [Model] U2NetP Active.")
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            sys.exit(1)

    # --- 数学与曲线辅助 ---
    def smooth_data(self, data, window_size, polyorder=3):
        data_len = len(data)
        if data_len < 4: return data 
        if window_size >= data_len: window_size = data_len - 1
        if window_size % 2 == 0: window_size -= 1
        if window_size < 3: return data
        if polyorder >= window_size: polyorder = window_size - 1
        return savgol_filter(data, window_size, polyorder)

    def linear_damp(self, t):
        return t

    def get_saliency_roi(self, img):
        saliency_map_small = self.saliency_detector.detect(img)
        saliency_map = cv2.resize(saliency_map_small, (img.shape[1], img.shape[0]))
        
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
            
            # 重心修正
            aspect_ratio = rh / rw if rw > 0 else 0
            if aspect_ratio > 1.2: 
                visual_cy = ry + (rh * 0.35) 
                visual_cx = phys_cx 
                final_cx, final_cy = visual_cx, visual_cy
            else:
                final_cx, final_cy = phys_cx, phys_cy
                
        return (final_cx, final_cy), saliency_ratio, saliency_map, rect, sal_top, sal_bottom

    def scan_video(self, clip):
        print(f"   [Phase 1] 扫描特征 | 持续时间: {clip.duration:.2f}s | FPS: {clip.fps}")
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
                
                raw_cx.append(cx); raw_cy.append(cy)
                raw_ratios.append(ratio); raw_rects.append(rect)
                raw_tops.append(s_top * scale_f); raw_bottoms.append(s_bot * scale_f)
            else:
                raw_cx.append(raw_cx[-1] if raw_cx else src_w/2)
                raw_cy.append(raw_cy[-1] if raw_cy else src_h/2)
                raw_ratios.append(raw_ratios[-1] if raw_ratios else 0.5)
                raw_rects.append(raw_rects[-1] if raw_rects else [0,0,0,0])
                raw_tops.append(raw_tops[-1] if raw_tops else 0)
                raw_bottoms.append(raw_bottoms[-1] if raw_bottoms else src_h)

        # 扫描数据平滑: 保留环绕时的运动趋势 (V8逻辑)
        win_pos = int(clip.fps * 1.0) 
        win_static = int(clip.fps * 4.0)
        
        smooth_cx = self.smooth_data(raw_cx, win_pos)
        smooth_cy = self.smooth_data(raw_cy, win_pos)
        smooth_ratios = self.smooth_data(raw_ratios, win_static)
        smooth_tops = self.smooth_data(raw_tops, win_pos)
        smooth_bottoms = self.smooth_data(raw_bottoms, win_pos)

        return {
            "smooth_cx": smooth_cx, "smooth_cy": smooth_cy,
            "smooth_ratios": smooth_ratios,
            "raw_rects": raw_rects, "raw_tops": raw_tops,
            "smooth_tops": smooth_tops, "smooth_bottoms": smooth_bottoms,
            "src_w": src_w, "src_h": src_h, "duration": clip.duration, "fps": clip.fps
        }

    # =======================================================
    # 核心路径求解器 (V8 Orbit-Aware Logic)
    # =======================================================

    def solve_crop_path(self, data, mode):
        """ 
        路径计算核心逻辑 V8 (Orbit Lock):
        1. Screen 3 (Hard Lock): 强制对齐主体中心，消除环绕时的离心力偏移。
        2. Screen 2 (Gimbal): 基于 Hard Lock 路径进行 SG 滤波平滑。
        3. Screen 1 (HUD): 视觉锚点 + 实时路径。
        """
        src_w, src_h = data["src_w"], data["src_h"]
        frames = len(data["smooth_cx"])
        fps = data["fps"]
        duration = data["duration"]
        
        gimbal_h_list, gimbal_cx_list, gimbal_cy_list = [], [], [] # Screen 2
        hard_h_list, hard_cx_list, hard_cy_list = [], [], []       # Screen 3
        hud_h_list, hud_cx_list, hud_cy_list = [], [], []          # Screen 1
        
        action_trace = []
        
        # 记录当前处理的视频文件名
        current_file = self.logger.current_video_log["file"] if hasattr(self.logger, "current_video_log") else "Unknown"
        print(f"   [Phase 2] 计算路径 | 模式: {mode} | 策略: Hard Center Lock")

        # --- 全局锚点 ---
        weights = np.array(data["smooth_ratios"]) + 0.01
        global_cx = np.average(data["smooth_cx"], weights=weights)
        global_cy = np.average(data["smooth_cy"], weights=weights)
        
        max_ratio = np.max(data["smooth_ratios"])
        global_target_scale = 0.5 / (max_ratio + 0.25)
        global_target_scale = max(1.1, min(global_target_scale, 1.6)) 

        avg_top = np.average(data["smooth_tops"], weights=weights)
        target_h_final = src_h / global_target_scale
        
        pan_end_x = global_cx
        pan_start_x = global_cx - (src_w * 0.15) if global_cx > src_w/2 else global_cx + (src_w * 0.15)
        if mode == "PAN": pan_end_x = global_cx

        frozen_h, frozen_cx, frozen_cy = src_h, src_w/2, src_h/2
        SHAKE_THRESHOLD = src_w * 0.02 

        for i in range(frames):
            curr_time = i / fps
            
            prog_gimbal = curr_time / (duration * 1.5)
            ease_gimbal = self.linear_damp(prog_gimbal)
            
            prog_hard = min(1.0, curr_time / duration)
            ease_hard = self.linear_damp(prog_hard)
            
            trace_msg = ""
            h_ref, cx_ref, cy_ref = 0,0,0    
            h_gimb, cx_gimb, cy_gimb = 0,0,0 
            h_hard, cx_hard, cy_hard = 0,0,0 

            if mode == "LOCKED_ON":
                # === V8 绝对中心锁死逻辑 ===
                s_cx = data["smooth_cx"][i]
                s_cy = data["smooth_cy"][i]
                s_ratio = data["smooth_ratios"][i]
                
                target_scale = 0.5 / (s_ratio + 0.25)
                target_scale = max(1.1, min(target_scale, 1.6))
                ideal_h = src_h / target_scale
                
                s_top = data["smooth_tops"][i]
                head_buf = ideal_h * 0.12
                ideal_cy = s_cy
                if (ideal_cy - ideal_h/2) > (s_top - head_buf):
                    ideal_cy = s_top - head_buf + ideal_h/2
                ideal_cx = s_cx

                # Screen 3: Hard Lock (Direct assignment)
                h_hard, cx_hard, cy_hard = ideal_h, ideal_cx, ideal_cy
                trace_msg = "LOCKED: ABSOLUTE"

                # Screen 2: Gimbal (Same as Hard, smoothed later)
                h_gimb, cx_gimb, cy_gimb = ideal_h, ideal_cx, ideal_cy
                
                # Screen 1: HUD Anchor (Freeze Logic)
                if i == 0:
                    frozen_h, frozen_cx, frozen_cy = ideal_h, ideal_cx, ideal_cy
                else:
                    dist = np.sqrt((ideal_cx - frozen_cx)**2 + (ideal_cy - frozen_cy)**2)
                    if dist > SHAKE_THRESHOLD:
                        frozen_h, frozen_cx, frozen_cy = ideal_h, ideal_cx, ideal_cy
                
                h_ref, cx_ref, cy_ref = frozen_h, frozen_cx, frozen_cy

            elif mode == "ZOOM_IN":
                h_ref, cx_ref, cy_ref = target_h_final, global_cx, global_cy
                
                h_hard = src_h + (target_h_final - src_h) * ease_hard
                z_ratio = (src_h - h_hard) / (src_h - target_h_final + 1e-6)
                cx_hard = (src_w/2) + (global_cx - src_w/2) * z_ratio
                cy_hard = (src_h/2) + (global_cy - src_h/2) * z_ratio
                
                h_gimb = src_h + (target_h_final - src_h) * ease_gimbal
                z_ratio_l = (src_h - h_gimb) / (src_h - target_h_final + 1e-6)
                cx_gimb = (src_w/2) + (global_cx - src_w/2) * z_ratio_l
                cy_gimb = (src_h/2) + (global_cy - src_h/2) * z_ratio_l
                
                trace_msg = f"ZOOM-IN {(ease_gimbal*100):.0f}%"

            elif mode == "ZOOM_OUT":
                h_ref, cx_ref, cy_ref = src_h, src_w/2, src_h/2
                
                h_hard = target_h_final + (src_h - target_h_final) * ease_hard
                z_ratio = (h_hard - target_h_final) / (src_h - target_h_final + 1e-6)
                cx_hard = global_cx + ((src_w/2) - global_cx) * z_ratio
                cy_hard = global_cy + ((src_h/2) - global_cy) * z_ratio
                
                h_gimb = target_h_final + (src_h - target_h_final) * ease_gimbal
                z_ratio_l = (h_gimb - target_h_final) / (src_h - target_h_final + 1e-6)
                cx_gimb = global_cx + ((src_w/2) - global_cx) * z_ratio_l
                cy_gimb = global_cy + ((src_h/2) - global_cy) * z_ratio_l

                trace_msg = f"ZOOM-OUT {(ease_gimbal*100):.0f}%"

            elif mode == "PAN":
                h_ref = target_h_final
                cy_ref = global_cy
                cx_ref = pan_end_x
                
                h_hard, cy_hard = target_h_final, global_cy
                cx_hard = pan_start_x + (pan_end_x - pan_start_x) * ease_hard
                
                h_gimb, cy_gimb = target_h_final, global_cy
                cx_gimb = pan_start_x + (pan_end_x - pan_start_x) * ease_gimbal
                
                trace_msg = "PANNING"
            
            else:
                h_ref, cx_ref, cy_ref = src_h, src_w/2, src_h/2
                h_hard, cx_hard, cy_hard = src_h, src_w/2, src_h/2
                h_gimb, cx_gimb, cy_gimb = src_h, src_w/2, src_h/2
                trace_msg = "IDLE"

            # Clamping
            def clamp_rect(ph, pcx, pcy):
                pw = ph * self.output_aspect
                if pw > src_w: pw = src_w; ph = pw / self.output_aspect
                if (pcx - pw/2) < 0: pcx = pw/2
                if (pcx + pw/2) > src_w: pcx = src_w - pw/2
                if (pcy - ph/2) < 0: pcy = ph/2
                if (pcy + ph/2) > src_h: pcy = src_h - ph/2
                return ph, pcx, pcy

            h_gimb, cx_gimb, cy_gimb = clamp_rect(h_gimb, cx_gimb, cy_gimb)
            h_hard, cx_hard, cy_hard = clamp_rect(h_hard, cx_hard, cy_hard)
            h_ref, cx_ref, cy_ref = clamp_rect(h_ref, cx_ref, cy_ref)

            gimbal_h_list.append(h_gimb); gimbal_cx_list.append(cx_gimb); gimbal_cy_list.append(cy_gimb)
            hard_h_list.append(h_hard); hard_cx_list.append(cx_hard); hard_cy_list.append(cy_hard)
            hud_h_list.append(h_ref); hud_cx_list.append(cx_ref); hud_cy_list.append(cy_ref)
            
            action_trace.append(trace_msg)
            if i % int(fps) == 0:
                self.logger.log_frame(i, curr_time, {"act": trace_msg})

        # --- 后处理 (Gimbal Smoothing) ---
        if mode == "LOCKED_ON":
            win = int(fps * 0.5) 
            if win % 2 == 0: win += 1
            if len(gimbal_h_list) > win:
                gimbal_h_list = savgol_filter(gimbal_h_list, win, 3)
                gimbal_cx_list = savgol_filter(gimbal_cx_list, win, 3)
                gimbal_cy_list = savgol_filter(gimbal_cy_list, win, 3)

        def to_rects(h_l, cx_l, cy_l):
            rects = []
            for i in range(frames):
                fh, fcx, fcy = h_l[i], cx_l[i], cy_l[i]
                fw = fh * self.output_aspect
                rects.append([fcx - fw/2, fcy - fh/2, fw, fh])
            return np.array(rects)

        gimbal_rects = to_rects(gimbal_h_list, gimbal_cx_list, gimbal_cy_list)
        hard_rects = to_rects(hard_h_list, hard_cx_list, hard_cy_list)
        hud_rects = to_rects(hud_h_list, hud_cx_list, hud_cy_list)
        
        return {
            "gimbal_rects": (gimbal_rects[:,0], gimbal_rects[:,1], gimbal_rects[:,2], gimbal_rects[:,3]),
            "hard_rects": (hard_rects[:,0], hard_rects[:,1], hard_rects[:,2], hard_rects[:,3]),
            "hud_rects": (hud_rects[:,0], hud_rects[:,1], hud_rects[:,2], hud_rects[:,3]),
            "action_trace": action_trace,
            "mode_name": mode,
            "scan_data": data
        }

    # =======================================================
    # 渲染器 (Triple Monitor)
    # =======================================================

    def render_hud_monitor(self, clip, result_packet):
        print(f"   [Phase 3] 渲染三屏监视器 ({result_packet['mode_name']})...")
        
        gx, gy, gw, gh = result_packet["gimbal_rects"]
        hx, hy, hw, hh = result_packet["hard_rects"]
        ref_x, ref_y, ref_w, ref_h = result_packet["hud_rects"]
        
        action_trace = result_packet["action_trace"]
        scan_data = result_packet["scan_data"]
        raw_rects = scan_data["raw_rects"]
        ratios = scan_data["smooth_ratios"]
        smooth_tops = scan_data["smooth_tops"]
        
        src_w, src_h = scan_data["src_w"], scan_data["src_h"]
        
        def frame_process(get_frame, t):
            raw_frame = get_frame(t)
            idx = min(int(t * clip.fps), len(gx)-1)
            
            # --- Screen 1: HUD ---
            small_viz = cv2.resize(raw_frame, (320, 180))
            _, _, s_map_small, _, _, _ = self.get_saliency_roi(small_viz)
            s_map_small = cv2.normalize(s_map_small, None, 0, 255, cv2.NORM_MINMAX)
            heatmap = cv2.applyColorMap(s_map_small, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (src_w, src_h))
            hud_bg = cv2.addWeighted(raw_frame, 0.60, heatmap, 0.40, 0)

            scale = self.target_h / src_h
            mon_w = int(src_w * scale)
            hud_img = cv2.resize(hud_bg, (mon_w, self.target_h))
            def to_mon(x, y): return int(x * scale), int(y * scale)

            if idx < len(smooth_tops):
                _, m_head_y = to_mon(0, smooth_tops[idx])
                cv2.line(hud_img, (0, m_head_y), (mon_w, m_head_y), (255, 0, 255), 1)

            # HUD Anchor
            rx1, ry1 = int(ref_x[idx]), int(ref_y[idx])
            rx2, ry2 = int(ref_x[idx]+ref_w[idx]), int(ref_y[idx]+ref_h[idx])
            mrx1, mry1 = to_mon(rx1, ry1); mrx2, mry2 = to_mon(rx2, ry2)
            draw_dashed_rect(hud_img, mrx1, mry1, mrx2, mry2, (0, 255, 255), 2)

            # Gimbal Path (Green)
            igx1, igy1 = int(gx[idx]), int(gy[idx])
            igx2, igy2 = int(gx[idx]+gw[idx]), int(gy[idx]+gh[idx])
            mgx1, mgy1 = to_mon(igx1, igy1); mgx2, mgy2 = to_mon(igx2, igy2)
            cv2.rectangle(hud_img, (mgx1, mgy1), (mgx2, mgy2), (0, 255, 0), 2)
            
            draw_vector_arrow(hud_img, ((mgx1+mgx2)//2, (mgy1+mgy2)//2), ((mrx1+mrx2)//2, (mry1+mry2)//2), (0, 255, 255))
            
            sx, sy, sw, sh = raw_rects[idx]
            bx1, by1 = to_mon(sx, sy); bx2, by2 = to_mon(sx+sw, sy+sh)
            draw_bracket(hud_img, bx1, by1, bx2, by2, (0, 0, 255), thickness=1)
            
            draw_hud_label(hud_img, "HUD: ANCHOR + RAW", 20, self.target_h - 40, bg_color=(50, 50, 50))
            draw_progress_bar(hud_img, 20, self.target_h - 70, 150, 10, ratios[idx], 0.6, (0, 200, 255), "SALIENCY")

            # --- Screen 2: Gimbal ---
            kx1, ky1 = max(0, igx1), max(0, igy1)
            kx2, ky2 = min(src_w, igx2), min(src_h, igy2)
            if kx2 <= kx1 or ky2 <= ky1: screen2_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            else:
                try: screen2_img = cv2.resize(raw_frame[ky1:ky2, kx1:kx2], (self.target_w, self.target_h))
                except: screen2_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            
            curr_zoom = src_h / gh[idx]
            label2 = "SCREEN 2: GIMBAL (SMOOTH)" if result_packet['mode_name'] == "LOCKED_ON" else "SCREEN 2: 66% PROGRESS"
            draw_hud_label(screen2_img, label2, 20, self.target_h - 40, bg_color=(0, 100, 0))
            draw_hud_label(screen2_img, f"ZOOM: {curr_zoom:.2f}x | {action_trace[idx]}", 20, self.target_h - 80, bg_color=(0, 100, 0))

            # --- Screen 3: Hard Lock ---
            ihx1, ihy1 = int(hx[idx]), int(hy[idx])
            ihx2, ihy2 = int(hx[idx]+hw[idx]), int(hy[idx]+hh[idx])
            jhx1, jhy1 = max(0, ihx1), max(0, ihy1)
            jhx2, jhy2 = min(src_w, ihx2), min(src_h, ihy2)
            
            if jhx2 <= jhx1 or jhy2 <= jhy1: screen3_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            else:
                try: screen3_img = cv2.resize(raw_frame[jhy1:jhy2, jhx1:jhx2], (self.target_w, self.target_h))
                except: screen3_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            
            hard_zoom = src_h / hh[idx]
            label3 = "SCREEN 3: HARD LOCK (ABSOLUTE)" if result_packet['mode_name'] == "LOCKED_ON" else "SCREEN 3: 100% PROGRESS"
            draw_hud_label(screen3_img, label3, 20, self.target_h - 40, bg_color=(0, 0, 100))
            draw_hud_label(screen3_img, f"ZOOM: {hard_zoom:.2f}x", 20, self.target_h - 80, bg_color=(0, 0, 100))

            # --- 拼接 ---
            total_w = self.target_w * 3
            combined = np.zeros((self.target_h, total_w, 3), dtype=np.uint8)
            
            off_x1 = (self.target_w - mon_w) // 2
            combined[:, off_x1:off_x1+mon_w] = hud_img
            combined[:, self.target_w:self.target_w*2] = screen2_img
            combined[:, self.target_w*2:] = screen3_img
            
            cv2.line(combined, (self.target_w, 0), (self.target_w, self.target_h), (255, 255, 255), 2)
            cv2.line(combined, (self.target_w*2, 0), (self.target_w*2, self.target_h), (255, 255, 255), 2)
            
            return combined

        return clip.fl(frame_process)

# ==========================================
# 模块 5: 封装功能函数 (Batch Support)
# ==========================================

_GLOBAL_ENGINE = None

def _get_engine():
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = CinematicEngine()
    return _GLOBAL_ENGINE

def _process_generic(video_path, mode, output_suffix):
    """ 处理单个视频的核心逻辑 """
    engine = _get_engine()
    if not os.path.exists(video_path):
        print(f"!!! 错误: 文件不存在 {video_path}")
        return

    print(f"\n>>> 正在处理: {os.path.basename(video_path)} | 模式: {mode}")
    
    # 注册到日志
    engine.logger.start_new_video(os.path.basename(video_path), mode)
    
    clip = VideoFileClip(video_path)
    scan_data = engine.scan_video(clip)
    result = engine.solve_crop_path(scan_data, mode=mode)
    final_clip = engine.render_hud_monitor(clip, result)
    
    out_dir = "Cinematic_Output_V9_Batch"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    fname = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{fname}_{output_suffix}_OrbitBatch.mp4")
    
    # 实时保存日志（防止崩溃丢失）
    log_path = os.path.join(out_dir, f"batch_log_{engine.logger.logs['batch_id']}.json")
    engine.logger.save_to_disk(log_path)
    
    print(f"    正在渲染至: {out_path}")
    # 渲染
    final_clip.write_videofile(out_path, codec='libx264', audio_codec='aac', fps=24, threads=4, logger='bar')
    
    clip.close()
    final_clip.close()
    print(f"√ {os.path.basename(video_path)} 处理完成\n")

# --- 支持传入多个路径的封装函数 ---

def zoom_in(*video_paths):
    total = len(video_paths)
    for i, path in enumerate(video_paths):
        print(f"--- Batch Progress: [{i+1}/{total}] ---")
        _process_generic(path, "ZOOM_IN", "ZoomIn")

def zoom_out(*video_paths):
    total = len(video_paths)
    for i, path in enumerate(video_paths):
        print(f"--- Batch Progress: [{i+1}/{total}] ---")
        _process_generic(path, "ZOOM_OUT", "ZoomOut")

def pan(*video_paths):
    total = len(video_paths)
    for i, path in enumerate(video_paths):
        print(f"--- Batch Progress: [{i+1}/{total}] ---")
        _process_generic(path, "PAN", "Pan")

def locked_on(*video_paths):
    total = len(video_paths)
    for i, path in enumerate(video_paths):
        print(f"--- Batch Progress: [{i+1}/{total}] ---")
        _process_generic(path, "LOCKED_ON", "LockedOn")


if __name__ == "__main__":
    print("="*60)
    print("【Cinematic Director - V9 Batch Processing】")
    print("Features: Batch Input | Single Engine Init | V8 Orbit Logic")
    print("="*60 + "\n")
    
    # 示例：批量处理您提供的所有视频
    locked_on(
        r"C:\Users\Administrator\Desktop\vlog-clean\data\wave1.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\wave2.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\wave3.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\red1.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\red2.mp4"
    )