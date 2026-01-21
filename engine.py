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
# 模块 1: 工具类与日志系统 (NumPy 2.0 兼容)
# ==========================================

class NumpyEncoder(json.JSONEncoder):
    """
    专门用于处理 NumPy 数据类型的 JSON 编码器
    解决 NumPy 2.0 兼容性问题
    """
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
            "video_file": "",
            "process_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "",
            "events": []
        }

    def set_file(self, filename, mode):
        self.logs["video_file"] = filename
        self.logs["mode"] = mode

    def log_frame(self, frame_idx, time_sec, params):
        entry = {
            "frame": int(frame_idx),
            "time": float(f"{time_sec:.3f}"),
            "params": params
        }
        self.logs["events"].append(entry)

    def save_to_disk(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        print(f"   [System] 运镜日志已保存至: {filepath}")

# ==========================================
# 模块 2: UI 绘图工具库 (Pro HUD) - 完整保留
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

def draw_vector_arrow(img, p1, p2, color=(0, 255, 255), thickness=4, tip_size=0.3):
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
        print(f"[System] 初始化 Cinematic Engine (集成 Tracking/Zoom/Pan)...")
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

    def ease_quintic_in_out(self, t): 
        if t < 0.5: return 16 * t * t * t * t * t
        else: return 1 - pow(-2 * t + 2, 5) / 2

    def ease_out_cubic(self, t):
        return 1 - pow(1 - t, 3)

    def ease_in_cubic(self, t):
        return t * t * t

    def get_saliency_roi(self, img):
        # 统一的视觉分析接口
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
            
            # 质心计算
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                phys_cx = int(M["m10"] / M["m00"])
                phys_cy = int(M["m01"] / M["m00"])
            else:
                phys_cx, phys_cy = rx + rw/2, ry + rh/2
            
            # V32 锚点优化：高瘦物体（如站立人）偏向视觉上方
            aspect_ratio = rh / rw if rw > 0 else 0
            if aspect_ratio > 1.2: 
                visual_cy = ry + (rh * 0.35) 
                visual_cx = phys_cx 
                final_cx, final_cy = visual_cx, visual_cy
            else:
                final_cx, final_cy = phys_cx, phys_cy
                
        return (final_cx, final_cy), saliency_ratio, saliency_map, rect, sal_top, sal_bottom

    def scan_video(self, clip):
        """ 第一阶段：全片扫描，提取原始特征数据 """
        print(f"   [Phase 1] 正在扫描视频特征 (Resolution: {clip.w}x{clip.h})...")
        src_w, src_h = clip.w, clip.h
        
        raw_cx, raw_cy, raw_ratios, raw_rects = [], [], [], []
        raw_tops, raw_bottoms = [], []
        
        # 降采样加速检测
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
                # 填充空缺帧
                raw_cx.append(raw_cx[-1] if raw_cx else src_w/2)
                raw_cy.append(raw_cy[-1] if raw_cy else src_h/2)
                raw_ratios.append(raw_ratios[-1] if raw_ratios else 0.5)
                raw_rects.append(raw_rects[-1] if raw_rects else [0,0,0,0])
                raw_tops.append(raw_tops[-1] if raw_tops else 0)
                raw_bottoms.append(raw_bottoms[-1] if raw_bottoms else src_h)

        # 基础数据平滑 (Savitzky-Golay)
        win_pos = int(clip.fps * 2.0)
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
    # 核心路径求解器
    # =======================================================

    def solve_crop_path(self, data, mode):
        """ 第二阶段：根据指定模式计算 Crop Window """
        src_w, src_h = data["src_w"], data["src_h"]
        frames = len(data["smooth_cx"])
        fps = data["fps"]
        
        t_h_list, t_cx_list, t_cy_list = [], [], []
        action_trace = []
        
        self.logger.set_file("Processing...", mode)
        print(f"   [Phase 2] 计算运镜路径 | 模式: {mode}")

        for i in range(frames):
            curr_time = i / fps
            prog = curr_time / data["duration"]
            
            # --- 基础数据获取 ---
            s_cx = data["smooth_cx"][i]
            s_cy = data["smooth_cy"][i]
            s_ratio = data["smooth_ratios"][i]
            s_top = data["smooth_tops"][i]
            s_bot = data["smooth_bottoms"][i]
            
            # 计算自然缩放比例 (Natural Scale)
            target_scale = 0.5 / (s_ratio + 0.25)
            hard_limit = 1.6
            target_scale = max(1.1, min(target_scale, hard_limit))
            
            # 基础目标框大小
            final_h_target = src_h / target_scale
            final_h_target = max(final_h_target, 480) # 最小分辨率保护
            
            # 基础目标位置 (带头部保护)
            final_cy_target = s_cy + (final_h_target * -0.05)
            head_buffer = final_h_target * 0.12
            if (final_cy_target - final_h_target/2) > (s_top - head_buffer):
                final_cy_target = s_top - head_buffer + final_h_target/2

            # ================= 模式分发逻辑 =================
            
            if mode == "LOCKED_ON":
                # === 锁定跟随模式 (V32 Tracking) ===
                # 特点：实时响应，始终保持主体在中心，缩放随主体大小变化
                frame_h = final_h_target
                frame_cx = s_cx
                frame_cy = final_cy_target
                
                trace_msg = "TRACKING"

            elif mode == "ZOOM_IN":
                # === 推进模式 (Zoom In) ===
                # 特点：全景 -> 特写，使用 Ease Out Cubic
                ease_val = self.ease_out_cubic(min(1.0, prog))
                
                # 高度插值：src_h -> final_h_target
                frame_h = src_h + (final_h_target - src_h) * ease_val
                
                # 中心点插值：画面中心 -> 主体中心
                frame_cx = (src_w/2) + (s_cx - src_w/2) * ease_val
                frame_cy = (src_h/2) + (final_cy_target - src_h/2) * ease_val
                
                trace_msg = f"ZOOM-IN {(ease_val*100):.0f}%"

            elif mode == "ZOOM_OUT":
                # === 拉远模式 (Zoom Out) ===
                # 特点：特写 -> 全景，使用 Ease In Cubic (先慢后快) 或 Quintic
                ease_val = self.ease_quintic_in_out(min(1.0, prog))
                
                # 高度插值：final_h_target -> src_h
                frame_h = final_h_target + (src_h - final_h_target) * ease_val
                
                # 中心点插值：主体中心 -> 画面中心
                frame_cx = s_cx + ((src_w/2) - s_cx) * ease_val
                frame_cy = final_cy_target + ((src_h/2) - final_cy_target) * ease_val
                
                trace_msg = f"ZOOM-OUT {(ease_val*100):.0f}%"

            elif mode == "PAN":
                # === 摇摄模式 (Pan) ===
                # 特点：保持特写景别，仅在 X 轴移动 (Ken Burns)
                # 策略：如果主体偏左，从右往左摇；反之亦然。或者简单地跟随主体 X，但保持 Y 和 Scale 相对稳定
                
                # 固定 Scale (取整个片段的中位数或平均值的 90%)
                avg_scale_h = np.mean([src_h / (0.5 / (r + 0.25)) for r in data["smooth_ratios"]])
                frame_h = max(avg_scale_h, src_h / 1.5) # 保持一定的特写感
                if frame_h > src_h: frame_h = src_h
                
                # Y 轴强稳定 (Highly Damped)
                frame_cy = final_cy_target # 初始值，稍后会由 SavGol 强平滑处理
                
                # X 轴跟随主体 (但在 Pan 模式下，我们会让它稍微滞后或平滑更多，由后续 smooth 处理)
                frame_cx = s_cx
                
                trace_msg = "PANNING"

            else:
                frame_h, frame_cx, frame_cy = src_h, src_w/2, src_h/2
                trace_msg = "IDLE"

            # === 物理边界限制 (Clamping) ===
            frame_w = frame_h * self.output_aspect
            
            # 如果目标比源还大，回退
            if frame_w > src_w: 
                frame_w = src_w
                frame_h = frame_w / self.output_aspect
            
            # 屏幕边缘限制
            if (frame_cx - frame_w/2) < 0: frame_cx = frame_w/2
            if (frame_cx + frame_w/2) > src_w: frame_cx = src_w - frame_w/2
            if (frame_cy - frame_h/2) < 0: frame_cy = frame_h/2
            if (frame_cy + frame_h/2) > src_h: frame_cy = src_h - frame_h/2

            t_h_list.append(frame_h)
            t_cx_list.append(frame_cx)
            t_cy_list.append(frame_cy)
            action_trace.append(trace_msg)
            
            # Log
            if i % int(fps) == 0:
                self.logger.log_frame(i, curr_time, {"cx": frame_cx, "cy": frame_cy, "h": frame_h, "act": trace_msg})

        # --- 最终路径平滑 (Post-Smoothing) ---
        # 不同的模式需要不同的平滑力度
        if mode == "LOCKED_ON":
            win = int(fps * 0.5) # 响应快
        elif mode == "PAN":
            win = int(fps * 4.0) # 极度平滑
        else:
            win = int(fps * 1.0) # 标准
            
        final_h = self.smooth_data(t_h_list, win, 2)
        final_cx = self.smooth_data(t_cx_list, win, 2)
        final_cy = self.smooth_data(t_cy_list, win, 2)
        
        # 生成 Crop Rects [x, y, w, h]
        crop_rects = []
        target_boxes = [] # 用于HUD显示目标点
        
        for i in range(frames):
            fh, fcx, fcy = final_h[i], final_cx[i], final_cy[i]
            fw = fh * self.output_aspect
            crop_rects.append([fcx - fw/2, fcy - fh/2, fw, fh])
            target_boxes.append((fcx, fcy, fw, fh))
            
        crop_rects = np.array(crop_rects)
        
        return {
            "crop_box": (crop_rects[:,0], crop_rects[:,1], crop_rects[:,2], crop_rects[:,3]),
            "target_box": target_boxes,
            "action_trace": action_trace,
            "mode_name": mode,
            "scan_data": data
        }

    # =======================================================
    # 渲染器 (HUD Monitor)
    # =======================================================

    def render_hud_monitor(self, clip, result_packet):
        print(f"   [Phase 3] 渲染 HUD 监视器 ({result_packet['mode_name']})...")
        cx, cy, w, h = result_packet["crop_box"]
        targets = result_packet["target_box"]
        action_trace = result_packet["action_trace"]
        scan_data = result_packet["scan_data"]
        
        raw_rects = scan_data["raw_rects"]
        ratios = scan_data["smooth_ratios"]
        smooth_tops = scan_data["smooth_tops"]
        smooth_bottoms = scan_data["smooth_bottoms"]
        
        src_w, src_h = scan_data["src_w"], scan_data["src_h"]
        
        def frame_process(get_frame, t):
            raw_frame = get_frame(t)
            idx = min(int(t * clip.fps), len(cx)-1)
            
            # 1. 显著性热力图生成 (V32 High Vis)
            small_viz = cv2.resize(raw_frame, (320, 180))
            _, _, s_map_small, _, _, _ = self.get_saliency_roi(small_viz)
            s_map_small = cv2.normalize(s_map_small, None, 0, 255, cv2.NORM_MINMAX) # 增强对比度
            heatmap = cv2.applyColorMap(s_map_small, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (src_w, src_h))
            
            # 混合背景：原图 60% + 热力图 40%
            monitor_bg = cv2.addWeighted(raw_frame, 0.60, heatmap, 0.40, 0)

            # 2. 最终输出图生成
            ix1, iy1 = int(cx[idx]), int(cy[idx])
            ix2, iy2 = int(cx[idx]+w[idx]), int(cy[idx]+h[idx])
            
            # 安全裁切
            rx1, ry1 = max(0, ix1), max(0, iy1)
            rx2, ry2 = min(src_w, ix2), min(src_h, iy2)
            if rx2 <= rx1 or ry2 <= ry1: 
                final_res_img = cv2.resize(raw_frame, (self.target_w, self.target_h))
            else:
                crop_img = raw_frame[ry1:ry2, rx1:rx2]
                try:
                    final_res_img = cv2.resize(crop_img, (self.target_w, self.target_h))
                except:
                    final_res_img = cv2.resize(raw_frame, (self.target_w, self.target_h))

            # 3. HUD 绘制
            scale = self.target_h / src_h
            mon_w = int(src_w * scale)
            mon_img = cv2.resize(monitor_bg, (mon_w, self.target_h))
            def to_mon(x, y): return int(x * scale), int(y * scale)
            
            # 辅助线
            if idx < len(smooth_tops):
                _, m_head_y = to_mon(0, smooth_tops[idx])
                cv2.line(mon_img, (0, m_head_y), (mon_w, m_head_y), (255, 0, 255), 1)
            
            # 目标框 (Target Box) - 黄色虚线
            tcx, tcy, tcw, tch = targets[idx]
            tx1, ty1 = int(tcx - tcw/2), int(tcy - tch/2)
            tx2, ty2 = int(tcx + tcw/2), int(tcy + tch/2)
            mtx1, mty1 = to_mon(tx1, ty1); mtx2, mty2 = to_mon(tx2, ty2)
            draw_dashed_rect(mon_img, mtx1, mty1, mtx2, mty2, (0, 255, 255), 1)
            
            # 当前裁切框 (Current Crop) - 绿色实线
            mx1, my1 = to_mon(ix1, iy1); mx2, my2 = to_mon(ix2, iy2)
            cv2.rectangle(mon_img, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            
            # 矢量箭头 (Vector)
            curr_center = ((mx1+mx2)//2, (my1+my2)//2)
            target_center = ((mtx1+mtx2)//2, (mty1+mty2)//2)
            draw_vector_arrow(mon_img, curr_center, target_center, (0, 255, 255))
            
            # 原始识别框 (Saliency Box) - 红色括号
            sx, sy, sw, sh = raw_rects[idx]
            bx1, by1 = to_mon(sx, sy); bx2, by2 = to_mon(sx+sw, sy+sh)
            draw_bracket(mon_img, bx1, by1, bx2, by2, (0, 0, 255), thickness=1)

            # 数据仪表盘
            ui_x, ui_y = 20, self.target_h - 60
            curr_zoom = src_h / h[idx]
            draw_hud_label(mon_img, f"ZOOM: {curr_zoom:.2f}x", ui_x, ui_y, bg_color=(0,100,0))
            draw_hud_label(mon_img, f"MODE: {result_packet['mode_name']}", ui_x + 130, ui_y, bg_color=(100, 0, 100))
            draw_hud_label(mon_img, f"ACT: {action_trace[idx]}", ui_x + 280, ui_y, bg_color=(50, 50, 50))
            
            draw_progress_bar(mon_img, ui_x, ui_y + 20, 150, 10, ratios[idx], 0.6, (0, 200, 255), "SALIENCY")
            draw_progress_bar(mon_img, ui_x + 170, ui_y + 20, 150, 10, curr_zoom, 2.0, (0, 255, 0), "SCALE")

            # 4. 左右拼接
            combined = np.zeros((self.target_h, self.target_w * 2, 3), dtype=np.uint8)
            off_x = (self.target_w - mon_w) // 2
            combined[:, off_x:off_x+mon_w] = mon_img
            combined[:, self.target_w:] = final_res_img
            
            return combined

        return clip.fl(frame_process)

# ==========================================
# 模块 5: 封装四大功能函数 (API)
# ==========================================

# 全局引擎实例 (懒加载模式)
_GLOBAL_ENGINE = None

def _get_engine():
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = CinematicEngine()
    return _GLOBAL_ENGINE

def _process_generic(video_path, mode, output_suffix):
    """ 通用处理流程封装 """
    engine = _get_engine()
    if not os.path.exists(video_path):
        print(f"!!! 错误: 文件不存在 {video_path}")
        return

    print(f"\n>>> 开始处理: {os.path.basename(video_path)} | 模式: {mode}")
    clip = VideoFileClip(video_path)
    
    # 1. 扫描与解算
    scan_data = engine.scan_video(clip)
    result = engine.solve_crop_path(scan_data, mode=mode)
    
    # 2. 渲染 HUD
    final_clip = engine.render_hud_monitor(clip, result)
    
    # 3. 导出
    out_dir = "Cinematic_Output"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    fname = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{fname}_{output_suffix}.mp4")
    
    # 保存日志
    log_path = os.path.join(out_dir, f"{fname}_{output_suffix}_log.json")
    engine.logger.save_to_disk(log_path)
    
    print(f"    正在渲染至: {out_path}")
    final_clip.write_videofile(out_path, codec='libx264', audio_codec='aac', fps=24, threads=4, logger='bar')
    
    clip.close()
    final_clip.close()
    print("√ 完成\n")

# --- 1. Zoom In 函数 ---
def zoom_in(video_path):
    """
    效果：从全景(1.0x)平滑推进到主体特写。
    适用：引入角色、强调细节。
    """
    _process_generic(video_path, "ZOOM_IN", "ZoomIn")

# --- 2. Zoom Out 函数 ---
def zoom_out(video_path):
    """
    效果：从主体特写拉远至全景。
    适用：结束镜头、展示环境关系。
    """
    _process_generic(video_path, "ZOOM_OUT", "ZoomOut")

# --- 3. Pan 函数 ---
def pan(video_path):
    """
    效果：保持相对固定的特写景别，水平跟随主体移动 (Ken Burns)。
    适用：横向运动物体、风景扫视。
    """
    _process_generic(video_path, "PAN", "Pan")

# --- 4. Locked On 函数 ---
def locked_on(video_path):
    """
    效果：高频追踪，始终将主体锁定在画面中心，模拟运动相机/跟拍。
    适用：舞蹈、演讲、高动态运动。
    """
    _process_generic(video_path, "LOCKED_ON", "LockedOn")


# ==========================================
# 主程序入口示例
# ==========================================

if __name__ == "__main__":
    print("="*60)
    print("【Cinematic Director API】")
    print("提供四种运镜模式: Zoom In, Zoom Out, Pan, Locked On")
    print("="*60 + "\n")
    
    if not os.path.exists("u2netp.onnx"):
        print("!!! 缺少 u2netp.onnx 模型，请确保文件在当前目录")
        sys.exit(1)
        
    # 测试视频路径 (请修改此处)
    TEST_VIDEO = r"C:\Users\Administrator\Desktop\vlog-clean\data\zoom_out.mp4"
    
    # 分别调用四个函数进行测试
    # 1. 锁定跟随
    # locked_on(TEST_VIDEO)
    
    # 2. 推进
    # zoom_in(TEST_VIDEO)
    
    # 3. 拉远
    zoom_out(TEST_VIDEO)
    
    # 4. 摇摄
    # pan(TEST_VIDEO)
    
    print("所有测试任务完成。")