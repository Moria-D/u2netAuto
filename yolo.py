import cv2
import numpy as np
import os
import sys
import time
import json
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter

# ==========================================
# [依赖检查] Ultralytics YOLO
# ==========================================
try:
    from ultralytics import YOLO
    print("[System] YOLOv8 库加载成功。")
except ImportError:
    print("!!! 错误: 请先安装 ultralytics 库。运行: pip install ultralytics")
    sys.exit(1)

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
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"   [System] 批次日志已保存至: {filepath}")

# ==========================================
# 模块 2: UI 绘图工具库
# ==========================================

def draw_hud_label(img, text, x, y, bg_color=(0, 0, 0), text_color=(255, 255, 255), font_scale=0.6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (t_w, t_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 5
    if y - t_h - pad < 0: y = t_h + pad + 5
    cv2.rectangle(img, (x, y - t_h - pad), (x + t_w + pad*2, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return x + t_w + pad*2 + 10

def draw_dashed_rect(img, x1, y1, x2, y2, color, thickness=1, dash_len=10):
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i+1)%4]
        if i % 2 == 0: 
            if p1[0] > p2[0]: p1, p2 = p2, p1
            for x in range(int(p1[0]), int(p2[0]), dash_len*2):
                cv2.line(img, (x, int(p1[1])), (min(x+dash_len, int(p2[0])), int(p1[1])), color, thickness)
        else: 
            if p1[1] > p2[1]: p1, p2 = p2, p1
            for y in range(int(p1[1]), int(p2[1]), dash_len*2):
                cv2.line(img, (int(p1[0]), y), (int(p1[0]), min(y+dash_len, int(p2[1]))), color, thickness)

def draw_progress_bar(img, x, y, w, h, val, max_val, color=(0, 255, 0), label=""):
    cv2.rectangle(img, (x, y), (x+w, y+h), (50, 50, 50), -1)
    ratio = max(0.0, min(1.0, val / max_val))
    fill_w = int(w * ratio)
    cv2.rectangle(img, (x, y), (x+fill_w, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (200, 200, 200), 1)
    if label:
        cv2.putText(img, f"{label}: {val:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# ==========================================
# 模块 3: 人物语义分割器 (YOLOv8-Seg Wrapper) - V22 升级
# ==========================================

class PersonSegmentor:
    def __init__(self, model_name='yolov8n-seg.pt'):
        print(f"   [Model] Loading YOLOv8 Segmentation ({model_name})...")
        # 这会自动下载分割模型
        self.model = YOLO(model_name)
        print("   [Model] Segmentation Model Loaded. Person Class ID: 0")

    def detect_and_segment(self, img_bgr):
        """
        返回: 
        1. (center_x, center_y)
        2. area_ratio
        3. bounding_box
        4. mask_img (二值化的人体掩膜, uint8 0-255)
        """
        h, w = img_bgr.shape[:2]
        # retina_masks=True 确保 mask 大小和原图一致，避免后续缩放锯齿
        results = self.model(img_bgr, verbose=False, classes=[0], conf=0.4, retina_masks=True)
        
        best_box = None
        best_area = 0
        best_mask = None
        
        for result in results:
            # 如果没检测到东西，result.boxes 可能有值但 result.masks 可能是 None
            if result.masks is None: continue
            
            boxes = result.boxes
            masks = result.masks.data # GPU tensor
            
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                bw = x2 - x1
                bh = y2 - y1
                area = bw * bh
                
                if area > best_area:
                    best_area = area
                    best_box = [int(x1), int(y1), int(bw), int(bh)]
                    # 提取对应的 mask，转为 numpy uint8
                    m = masks[i].cpu().numpy()
                    best_mask = (m * 255).astype(np.uint8)

        if best_box:
            bx, by, bw, bh = best_box
            cx = bx + bw / 2
            cy = by + bh / 2
            ratio = best_area / (w * h)
            
            # 确保 mask 尺寸匹配原图 (虽然 retina_masks=True 应该保证这点，但防万一)
            if best_mask is not None and best_mask.shape != (h, w):
                best_mask = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
            return (cx, cy), ratio, best_box, best_mask
        else:
            return None, 0.0, None, None

# ==========================================
# 模块 3.5: 视频防抖引擎
# ==========================================

class VideoStabilizer:
    def __init__(self):
        self.transforms = []
        self.smoothing_radius = 30 
        self.border_zoom = 1.05    

    def compute_transforms(self, clip):
        print("   [Phase 0] 正在分析视频抖动 (Optical Flow)...")
        prev_gray = None
        transforms = []
        frames_iter = list(clip.iter_frames())
        total_frames = len(frames_iter)
        last_t = [0, 0, 0] 
        
        for i, curr_frame in enumerate(frames_iter):
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            if prev_gray is None:
                prev_gray = curr_gray
                transforms.append([0, 0, 0])
                continue
            
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
            if prev_pts is not None:
                curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                idx = np.where(status==1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                if len(prev_pts) > 10: 
                    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                    if m is not None:
                        dx = m[0, 2]; dy = m[1, 2]
                        da = np.arctan2(m[1, 0], m[0, 0])
                        last_t = [dx, dy, da]
                    else: last_t = [0, 0, 0]
                else: last_t = [0, 0, 0]
            else: last_t = [0, 0, 0]
            transforms.append(last_t)
            prev_gray = curr_gray
            if i % 48 == 0:
                sys.stdout.write(f"\r   ... 分析进度: {int(i/total_frames*100)}%")
                sys.stdout.flush()
        print("\r   ... 抖动分析完成。               ")
        
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self.smooth_trajectory(trajectory)
        difference = smoothed_trajectory - trajectory
        self.transforms = difference
        return self.transforms

    def smooth_trajectory(self, trajectory):
        smoothed = np.copy(trajectory)
        window = self.smoothing_radius
        for i in range(3):
            smoothed[:, i] = self.moving_average(trajectory[:, i], radius=window)
        return smoothed

    def moving_average(self, curve, radius):
        window_size = 2 * radius + 1
        filter = np.ones(window_size) / window_size
        curve_padded = np.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_padded, filter, mode='same')
        return curve_smoothed[radius:-radius]

    def stabilize_frame(self, frame, idx):
        if idx >= len(self.transforms): return frame
        dx, dy, da = self.transforms[idx]
        h, w = frame.shape[:2]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da); m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da); m[1, 1] = np.cos(da)
        m[0, 2] = dx; m[1, 2] = dy
        scale = self.border_zoom
        T = np.array([[1, 0, -w/2], [0, 1, -h/2], [0, 0, 1]]) 
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]) 
        Ti = np.array([[1, 0, w/2], [0, 1, h/2], [0, 0, 1]]) 
        M_stab = np.vstack([m, [0, 0, 1]])
        M_final = Ti @ S @ T @ M_stab
        m_final = M_final[:2, :]
        return cv2.warpAffine(frame, m_final, (w, h), borderMode=cv2.BORDER_REFLECT)

# ==========================================
# 模块 4: 核心算法引擎 (CinematicEngine V22 - Segmentation)
# ==========================================

class CinematicEngine:
    def __init__(self, target_res=(1920, 1080)):
        print(f"[System] 初始化 Cinematic Engine (V22 Segmentation Heatmap)...")
        self.target_w, self.target_h = target_res
        self.output_aspect = self.target_w / self.target_h
        self.logger = EventLogger()
        self.stabilizer = VideoStabilizer() 
        
        try:
            # 升级为 yolov8n-seg.pt
            self.detector = PersonSegmentor('yolov8n-seg.pt') 
            print("   [Engine] 语义分割核心 (YOLOv8-Seg) 已启动")
        except Exception as e:
            print(f"!!! 模型加载失败: {e}")
            sys.exit(1)

    # --- 数学与曲线辅助 ---
    def smooth_data(self, data, window_size, polyorder=2):
        data_len = len(data)
        if data_len < 4: return data 
        if window_size >= data_len: window_size = data_len - 1
        if window_size % 2 == 0: window_size -= 1
        if window_size < 3: return data
        if polyorder >= window_size: polyorder = window_size - 1
        return savgol_filter(data, window_size, polyorder)

    def linear_damp(self, t):
        return t

    def scan_video(self, clip):
        print(f"   [Phase 1] 语义扫描 (YOLOv8-Seg) | Duration: {clip.duration:.2f}s")
        src_w, src_h = clip.w, clip.h
        
        self.stabilizer.compute_transforms(clip)
        
        raw_cx, raw_cy, raw_ratios, raw_rects = [], [], [], []
        raw_tops, raw_bottoms = [], []
        # 新增：保存 mask 的压缩版本 (resize 到小图以省内存，因为我们要渲染热力图)
        self.mask_cache = [] 
        
        last_valid_cx = src_w / 2
        last_valid_cy = src_h / 2
        last_valid_rect = [0, 0, 0, 0]
        last_valid_ratio = 0.2
        last_valid_top = 0
        last_valid_bot = src_h
        last_valid_mask = None

        detection_lost_counter = 0

        for i, frame in enumerate(clip.iter_frames()):
            stable_frame = self.stabilizer.stabilize_frame(frame, i)
            # 使用分割检测
            center, ratio, box, mask = self.detector.detect_and_segment(stable_frame)
            
            if box is not None:
                detection_lost_counter = 0
                cx, cy = center
                bx, by, bw, bh = box
                
                last_valid_cx = cx
                last_valid_cy = cy
                last_valid_rect = box
                last_valid_ratio = ratio
                last_valid_top = by
                last_valid_bot = by + bh
                
                # 压缩 Mask 以备 Phase 3 使用 (渲染分辨率)
                # 存成 640x360 足够渲染热力图了
                mask_small = cv2.resize(mask, (640, 360), interpolation=cv2.INTER_NEAREST)
                last_valid_mask = mask_small
                
                raw_cx.append(cx)
                raw_cy.append(cy)
                raw_ratios.append(ratio)
                raw_rects.append(box)
                raw_tops.append(by)
                raw_bottoms.append(by + bh)
                self.mask_cache.append(mask_small)
            else:
                detection_lost_counter += 1
                raw_cx.append(last_valid_cx)
                raw_cy.append(last_valid_cy)
                raw_ratios.append(last_valid_ratio)
                raw_rects.append(last_valid_rect)
                raw_tops.append(last_valid_top)
                raw_bottoms.append(last_valid_bot)
                self.mask_cache.append(last_valid_mask) # 保持上一帧的mask
            
            if i % 24 == 0:
                sys.stdout.write(f"\r   ... Scanning Frame {i} (Lost: {detection_lost_counter})")
                sys.stdout.flush()
        
        print("\n   [Phase 1] 扫描完成。开始轨迹平滑...")

        win_pos = int(clip.fps * 2.0) 
        win_zoom = int(clip.fps * 4.0) 
        
        smooth_cx = self.smooth_data(raw_cx, win_pos, polyorder=2)
        smooth_cy = self.smooth_data(raw_cy, win_pos, polyorder=2)
        smooth_ratios = self.smooth_data(raw_ratios, win_zoom, polyorder=1) 
        smooth_tops = self.smooth_data(raw_tops, win_pos, polyorder=2)
        smooth_bottoms = self.smooth_data(raw_bottoms, win_pos, polyorder=2)

        return {
            "smooth_cx": smooth_cx, "smooth_cy": smooth_cy,
            "smooth_ratios": smooth_ratios,
            "raw_rects": raw_rects, 
            "smooth_tops": smooth_tops, 
            "smooth_bottoms": smooth_bottoms,
            "src_w": src_w, "src_h": src_h, "duration": clip.duration, "fps": clip.fps
        }

    # =======================================================
    # 核心路径求解器
    # =======================================================

    def solve_crop_path(self, data, mode):
        src_w, src_h = data["src_w"], data["src_h"]
        frames = len(data["smooth_cx"])
        fps = data["fps"]
        duration = data["duration"]
        
        gimbal_h_list, gimbal_cx_list, gimbal_cy_list = [], [], [] 
        hard_h_list, hard_cx_list, hard_cy_list = [], [], []       
        hud_h_list, hud_cx_list, hud_cy_list = [], [], []          
        action_trace = []
        
        print(f"   [Phase 2] 计算路径 | 模式: {mode}")

        current_v_bias = 0.0 

        weights = np.array(data["smooth_ratios"]) + 0.01
        global_cx = np.average(data["smooth_cx"], weights=weights)
        
        max_ratio = np.max(data["smooth_ratios"])
        global_target_scale = 1.0 / (max_ratio + 0.3) 
        global_target_scale = max(1.1, min(global_target_scale, 2.0)) 
        
        target_h_final = src_h / global_target_scale

        avg_top = np.average(data["smooth_tops"], weights=weights)
        desired_head_buffer = target_h_final * 0.15 
        
        global_cy = avg_top + (target_h_final / 2) - desired_head_buffer
        if global_cy + target_h_final/2 > src_h: global_cy = src_h - target_h_final/2
        if global_cy - target_h_final/2 < 0: global_cy = target_h_final/2
        
        pan_end_x = global_cx
        pan_start_x = global_cx - (src_w * 0.15) if global_cx > src_w/2 else global_cx + (src_w * 0.15)
        if mode == "PAN": pan_end_x = global_cx

        frozen_h, frozen_cx, frozen_cy = src_h, src_w/2, src_h/2
        SHAKE_THRESHOLD = src_w * 0.005 

        def adjust_for_headroom_and_bias(curr_h, curr_cx, curr_geo_cy, subj_top, bias_ratio):
            biased_cy = curr_geo_cy 
            min_buffer = curr_h * 0.12 
            limit_top = subj_top - min_buffer
            if (biased_cy - curr_h/2) > limit_top:
                final_cy = limit_top + (curr_h / 2)
            else:
                final_cy = biased_cy 
            if (final_cy - curr_h/2) < 0: final_cy = curr_h / 2
            if (final_cy + curr_h/2) > src_h: final_cy = src_h - curr_h / 2
            return curr_h, curr_cx, final_cy

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

            s_top = data["smooth_tops"][i]
            s_cx = data["smooth_cx"][i]
            s_geo_cy = data["smooth_cy"][i]
            
            if mode == "LOCKED_ON":
                s_ratio = data["smooth_ratios"][i]
                target_scale = 1.0 / (s_ratio + 0.3)
                target_scale = max(1.1, min(target_scale, 2.0))
                ideal_h = src_h / target_scale
                
                ideal_h, ideal_cx, ideal_cy = adjust_for_headroom_and_bias(ideal_h, s_cx, s_geo_cy, s_top, current_v_bias)

                h_hard, cx_hard, cy_hard = ideal_h, ideal_cx, ideal_cy
                h_gimb, cx_gimb, cy_gimb = ideal_h, ideal_cx, ideal_cy
                
                if i == 0:
                    frozen_h, frozen_cx, frozen_cy = ideal_h, ideal_cx, ideal_cy
                else:
                    dist = np.sqrt((ideal_cx - frozen_cx)**2 + (ideal_cy - frozen_cy)**2)
                    size_diff = abs(ideal_h - frozen_h)
                    
                    if dist > SHAKE_THRESHOLD:
                        frozen_cx += (ideal_cx - frozen_cx) * 0.05
                        frozen_cy += (ideal_cy - frozen_cy) * 0.05
                    
                    if size_diff > (src_h * 0.05):
                         frozen_h += (ideal_h - frozen_h) * 0.02 
                    
                h_ref, cx_ref, cy_ref = frozen_h, frozen_cx, frozen_cy
                h_gimb, cx_gimb, cy_gimb = frozen_h, frozen_cx, frozen_cy
                trace_msg = "LOCKED (SEMANTIC)"

            elif mode == "ZOOM_IN":
                h_ref, cx_ref, cy_ref = target_h_final, global_cx, global_cy
                h_hard = src_h + (target_h_final - src_h) * ease_hard
                z_ratio = (src_h - h_hard) / (src_h - target_h_final + 1e-6)
                cx_hard = (src_w/2) + (global_cx - src_w/2) * z_ratio
                cy_hard_raw = (src_h/2) + (global_cy - src_h/2) * z_ratio
                h_hard, cx_hard, cy_hard = adjust_for_headroom_and_bias(h_hard, cx_hard, cy_hard_raw, s_top, 0.0) 
                
                h_gimb = src_h + (target_h_final - src_h) * ease_gimbal
                z_ratio_l = (src_h - h_gimb) / (src_h - target_h_final + 1e-6)
                cx_gimb = (src_w/2) + (global_cx - src_w/2) * z_ratio_l
                cy_gimb_raw = (src_h/2) + (global_cy - src_h/2) * z_ratio_l
                h_gimb, cx_gimb, cy_gimb = adjust_for_headroom_and_bias(h_gimb, cx_gimb, cy_gimb_raw, s_top, 0.0)
                trace_msg = f"ZOOM-IN {(ease_gimbal*100):.0f}%"

            elif mode == "ZOOM_OUT":
                h_ref, cx_ref, cy_ref = src_h, src_w/2, src_h/2
                h_hard = target_h_final + (src_h - target_h_final) * ease_hard
                z_ratio = (h_hard - target_h_final) / (src_h - target_h_final + 1e-6)
                cx_hard = global_cx + ((src_w/2) - global_cx) * z_ratio
                cy_hard_raw = global_cy + ((src_h/2) - global_cy) * z_ratio
                h_hard, cx_hard, cy_hard = adjust_for_headroom_and_bias(h_hard, cx_hard, cy_hard_raw, s_top, 0.0)
                
                h_gimb = target_h_final + (src_h - target_h_final) * ease_gimbal
                z_ratio_l = (h_gimb - target_h_final) / (src_h - target_h_final + 1e-6)
                cx_gimb = global_cx + ((src_w/2) - global_cx) * z_ratio_l
                cy_gimb_raw = global_cy + ((src_h/2) - global_cy) * z_ratio_l
                h_gimb, cx_gimb, cy_gimb = adjust_for_headroom_and_bias(h_gimb, cx_gimb, cy_gimb_raw, s_top, 0.0)
                trace_msg = f"ZOOM-OUT {(ease_gimbal*100):.0f}%"

            elif mode == "PAN":
                h_base = target_h_final
                cy_base = global_cy 
                h_base, _, cy_base = adjust_for_headroom_and_bias(h_base, 0, cy_base, s_top, 0.0)
                h_ref = h_base; cy_ref = cy_base; cx_ref = pan_end_x
                h_hard, cy_hard = h_base, cy_base
                cx_hard = pan_start_x + (pan_end_x - pan_start_x) * ease_hard
                h_gimb, cy_gimb = h_base, cy_base
                cx_gimb = pan_start_x + (pan_end_x - pan_start_x) * ease_gimbal
                trace_msg = "PANNING"
            
            else:
                h_ref, cx_ref, cy_ref = src_h, src_w/2, src_h/2
                h_hard, cx_hard, cy_hard = src_h, src_w/2, src_h/2
                h_gimb, cx_gimb, cy_gimb = src_h, src_w/2, src_h/2
                trace_msg = "IDLE"

            def clamp_rect(ph, pcx, pcy):
                pw = ph * self.output_aspect
                if pw > src_w: pw = src_w; ph = pw / self.output_aspect
                if (pcx - pw/2) < 0: pcx = pw/2
                if (pcx + pw/2) > src_w: pcx = src_w - pw/2
                if (pcy - ph/2) < 0: pcy = ph/2
                if (pcy + ph/2) > src_h: pcy = src_h - ph/2
                return ph, pcx, pcy

            h_gimb, cx_gimb, cy_gimb = clamp_rect(h_gimb, cx_gimb, cy_gimb)
            hard_h_list.append(h_hard); hard_cx_list.append(cx_hard); hard_cy_list.append(cy_hard)
            gimbal_h_list.append(h_gimb); gimbal_cx_list.append(cx_gimb); gimbal_cy_list.append(cy_gimb)
            hud_h_list.append(h_ref); hud_cx_list.append(cx_ref); hud_cy_list.append(cy_ref)
            
            action_trace.append(trace_msg)
            if i % int(fps) == 0:
                self.logger.log_frame(i, curr_time, {"act": trace_msg})

        if mode == "LOCKED_ON":
            win = int(fps * 1.5) 
            if win % 2 == 0: win += 1
            if len(gimbal_h_list) > win:
                gimbal_h_list = savgol_filter(gimbal_h_list, win, 2)
                gimbal_cx_list = savgol_filter(gimbal_cx_list, win, 2)
                gimbal_cy_list = savgol_filter(gimbal_cy_list, win, 2)

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
    # 渲染器 - [V22 升级] 分割掩膜热力图
    # =======================================================

    def render_hud_monitor(self, clip, result_packet):
        print(f"   [Phase 3] 渲染双屏监视器 (HUD + Result)...")
        gx, gy, gw, gh = result_packet["gimbal_rects"]
        hx, hy, hw, hh = result_packet["hard_rects"]
        ref_x, ref_y, ref_w, ref_h = result_packet["hud_rects"]
        action_trace = result_packet["action_trace"]
        scan_data = result_packet["scan_data"]
        ratios = scan_data["smooth_ratios"]
        smooth_tops = scan_data["smooth_tops"]
        src_w, src_h = scan_data["src_w"], scan_data["src_h"]
        
        # 重要的 mask 缓存
        mask_cache = self.mask_cache
        
        def frame_process(get_frame, t):
            raw_frame_orig = get_frame(t)
            idx = min(int(t * clip.fps), len(gx)-1)
            raw_frame = self.stabilizer.stabilize_frame(raw_frame_orig, idx)
            
            scale = min(self.target_w / src_w, self.target_h / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            
            # --- Screen 1: HUD with True Segmentation Heatmap ---
            heatmap_overlay = np.zeros_like(raw_frame, dtype=np.uint8)
            
            if idx < len(mask_cache):
                mask_data = mask_cache[idx]
                if mask_data is not None:
                    # 1. 把 640x360 的小 mask 放大回原图尺寸
                    mask_full = cv2.resize(mask_data, (raw_frame.shape[1], raw_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    # 2. 强力高斯模糊 (模拟热力辐射扩散)
                    mask_blur = cv2.GaussianBlur(mask_full, (101, 101), 0)
                    
                    # 3. 伪彩色映射
                    heatmap_color = cv2.applyColorMap(mask_blur, cv2.COLORMAP_JET)
                    
                    # 4. 掩膜裁剪 (只显示人体附近的热力，背景保持黑色)
                    # 阈值处理，让热力图边缘有淡出效果
                    _, mask_thresh = cv2.threshold(mask_blur, 10, 255, cv2.THRESH_BINARY)
                    heatmap_overlay = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask_thresh)

            # 缩放 Frame 和 Heatmap
            frame_resized = cv2.resize(raw_frame, (new_w, new_h))
            heatmap_resized = cv2.resize(heatmap_overlay, (new_w, new_h))
            
            # 叠加: Frame + Heatmap (Alpha 0.3)
            hud_blended = cv2.addWeighted(frame_resized, 1.0, heatmap_resized, 0.3, 0)
            
            # 放入画布
            hud_canvas = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)
            offset_x = (self.target_w - new_w) // 2
            offset_y = (self.target_h - new_h) // 2
            hud_canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = hud_blended
            
            def to_mon(x, y):
                nx = int(x * scale) + offset_x
                ny = int(y * scale) + offset_y
                return nx, ny

            if idx < len(smooth_tops):
                _, m_head_y = to_mon(0, smooth_tops[idx])
                cv2.line(hud_canvas, (offset_x, m_head_y), (offset_x+new_w, m_head_y), (0, 0, 255), 2)
            
            rx1, ry1 = int(ref_x[idx]), int(ref_y[idx])
            rx2, ry2 = int(ref_x[idx]+ref_w[idx]), int(ref_y[idx]+ref_h[idx])
            mrx1, mry1 = to_mon(rx1, ry1); mrx2, mry2 = to_mon(rx2, ry2)
            draw_dashed_rect(hud_canvas, mrx1, mry1, mrx2, mry2, (0, 255, 255), 2)

            igx1, igy1 = int(gx[idx]), int(gy[idx])
            igx2, igy2 = int(gx[idx]+gw[idx]), int(gy[idx]+gh[idx])
            mgx1, mgy1 = to_mon(igx1, igy1); mgx2, mgy2 = to_mon(igx2, igy2)
            cv2.rectangle(hud_canvas, (mgx1, mgy1), (mgx2, mgy2), (0, 255, 0), 2)
            
            draw_hud_label(hud_canvas, "HUD: YOLOv8-SEG HEATMAP", 20, self.target_h - 40, bg_color=(50, 50, 50))
            draw_progress_bar(hud_canvas, 20, self.target_h - 70, 150, 10, ratios[idx], 0.6, (0, 200, 255), "SUBJ SCALE")

            # --- Screen 2: Result ---
            ihx1, ihy1 = int(hx[idx]), int(hy[idx])
            ihx2, ihy2 = int(hx[idx]+hw[idx]), int(hy[idx]+hh[idx])
            jhx1, jhy1 = max(0, ihx1), max(0, ihy1)
            jhx2, jhy2 = min(src_w, ihx2), min(src_h, ihy2)
            
            if jhx2 <= jhx1 or jhy2 <= jhy1: screen3_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            else:
                try: 
                    crop = raw_frame[jhy1:jhy2, jhx1:jhx2]
                    if crop.size == 0: raise ValueError("Empty crop")
                    screen3_img = cv2.resize(crop, (self.target_w, self.target_h))
                except: screen3_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            
            hard_zoom = src_h / (hh[idx] + 1e-6)
            draw_hud_label(screen3_img, "RESULT: LOCKED ON", 20, self.target_h - 40, bg_color=(0, 0, 100))
            draw_hud_label(screen3_img, f"ZOOM: {hard_zoom:.2f}x", 20, self.target_h - 80, bg_color=(0, 0, 100))

            combined = np.zeros((self.target_h, self.target_w * 2, 3), dtype=np.uint8)
            combined[:, 0:self.target_w] = hud_canvas
            combined[:, self.target_w:] = screen3_img
            cv2.line(combined, (self.target_w, 0), (self.target_w, self.target_h), (255, 255, 255), 2)
            
            return combined

        return clip.fl(frame_process)

# ==========================================
# 模块 5: API & Batch
# ==========================================

_GLOBAL_ENGINE = None

def _get_engine():
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = CinematicEngine()
    return _GLOBAL_ENGINE

def _process_generic(video_path, mode, output_suffix):
    engine = _get_engine()
    if not os.path.exists(video_path):
        print(f"!!! Error: File not found {video_path}")
        return

    print(f"\n>>> Processing: {os.path.basename(video_path)} | Mode: {mode}")
    engine.logger.start_new_video(os.path.basename(video_path), mode)
    
    clip = VideoFileClip(video_path)
    scan_data = engine.scan_video(clip)
    result = engine.solve_crop_path(scan_data, mode=mode)
    final_clip = engine.render_hud_monitor(clip, result)
    
    out_dir = "Cinematic_Output_V22_Seg"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    fname = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{fname}_{output_suffix}_V22.mp4")
    
    log_path = os.path.join(out_dir, f"batch_log_{engine.logger.logs['batch_id']}.json")
    engine.logger.save_to_disk(log_path)
    
    print(f"    Rendering to: {out_path}")
    final_clip.write_videofile(out_path, codec='libx264', audio_codec='aac', fps=24, threads=4, logger='bar')
    
    clip.close()
    final_clip.close()
    print(f"√ Finished {os.path.basename(video_path)}\n")

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
    print("【Cinematic Director - V22 Segmentation Heatmap】")
    print("1. Upgraded to YOLOv8-Seg for precise body-shaped masks.")
    print("2. Visual: True heatmap generated from body segmentation.")
    print("3. Layout: Dual Screen (Left: Heatmap HUD | Right: Final).")
    print("="*60 + "\n")
    
    locked_on(
       r"C:\Users\Administrator\Desktop\vlog-clean\data\dance.mp4",
       r"C:\Users\Administrator\Desktop\vlog-clean\data\stage.mp4",
       r"C:\Users\Administrator\Desktop\vlog-clean\data\red1.mp4",
       r"C:\Users\Administrator\Desktop\vlog-clean\data\red2.mp4",
       r"C:\Users\Administrator\Desktop\vlog-clean\data\wave3.mp4",
       r"C:\Users\Administrator\Desktop\vlog-clean\data\wave2.mp4"
    )