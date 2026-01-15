import cv2
import numpy as np
import os
import sys
import time
import onnxruntime as ort
from moviepy.editor import VideoFileClip
from scipy.signal import savgol_filter

# ==========================================
# UI 绘图工具库
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

def draw_vector_arrow(img, p1, p2, color=(0, 255, 255), size=2):
    p1 = (int(p1[0]), int(p1[1])); p2 = (int(p2[0]), int(p2[1]))
    dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if dist > 2:
        cv2.arrowedLine(img, p1, p2, color, size, tipLength=0.3)

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
# 核心算法引擎 V33.2 (Bugfix)
# ==========================================

class DirectorEngine:
    def __init__(self, target_res=(1920, 1080)):
        print(f"[System] 初始化导演引擎 V33.2 (Bugfix)...")
        self.target_w, self.target_h = target_res
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
        # 1. U2Net
        saliency_map_small = self.saliency_detector.detect(img)
        saliency_map = cv2.resize(saliency_map_small, (img.shape[1], img.shape[0]))
        
        # 2. Mask
        _, thresh = cv2.threshold(saliency_map, 100, 255, cv2.THRESH_BINARY)
        white_pixels = cv2.countNonZero(thresh)
        saliency_ratio = white_pixels / (img.shape[0] * img.shape[1])
        
        # 3. Contour & Rect
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
            
            # Smart Anchor
            M = cv2.moments(main_contour)
            if M["m00"] != 0:
                phys_cx = int(M["m10"] / M["m00"])
                phys_cy = int(M["m01"] / M["m00"])
            else:
                phys_cx, phys_cy = rx + rw/2, ry + rh/2
                
            aspect_ratio = rh / rw if rw > 0 else 0
            if aspect_ratio > 1.2: 
                # Humanoid
                visual_cy = ry + (rh * 0.35) 
                visual_cx = phys_cx 
                final_cx, final_cy = visual_cx, visual_cy
            else:
                final_cx, final_cy = phys_cx, phys_cy
                
        return (final_cx, final_cy), saliency_ratio, saliency_map, rect, sal_top, sal_bottom

    def calculate_natural_scale(self, ratio, duration):
        """ V32 自然缩放逻辑 """
        target_scale = 0.5 / (ratio + 0.25)
        duration_limit = 1.0 + (duration * 0.08) 
        hard_limit = 1.6
        final_limit = min(duration_limit, hard_limit)
        target_scale = max(1.1, min(target_scale, final_limit))
        return target_scale

    def ease_quintic_in_out(self, t):
        if t < 0.5: return 16 * t * t * t * t * t
        else: return 1 - pow(-2 * t + 2, 5) / 2

    def analyze_content(self, clip):
        print(f"   [Phase 1] 扫描全片 (V33 Pipeline)...")
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

        if len(raw_cx) > 10:
            dx = np.diff(raw_cx); dy = np.diff(raw_cy)
            motion_score = np.mean(np.sqrt(dx**2 + dy**2))
        else: motion_score = 0
        
        mode = "DOLLY"
        if motion_score > 15.0: mode = "LOCKED-ON"
        need_slowmo = motion_score > 8.0 and clip.duration > 4

        # Stage 1: Input Smoothing
        win_pos = int(clip.fps * 3.0)
        win_ratio = int(clip.fps * 4.0)

        smooth_cx = self.smooth_data(raw_cx, win_pos)
        smooth_cy = self.smooth_data(raw_cy, win_pos)
        smooth_ratios = self.smooth_data(raw_ratios, win_ratio)
        smooth_tops = self.smooth_data(raw_tops, win_pos)
        smooth_bottoms = self.smooth_data(raw_bottoms, win_pos)
        
        # Stage 2: Calculate Instant Targets
        inst_target_h = []
        inst_target_cy = []
        inst_target_cx = []
        
        output_aspect = self.target_w / self.target_h
        
        for i in range(len(smooth_cx)):
            curr_ratio = smooth_ratios[i]
            safe_scale = self.calculate_natural_scale(curr_ratio, clip.duration)
            
            t_h = src_h / safe_scale
            t_h = max(t_h, 480) 
            
            # Vertical Fit (V30 Logic)
            curr_top = smooth_tops[i]
            curr_bottom = smooth_bottoms[i]
            full_height = curr_bottom - curr_top
            essential_bottom = curr_top + (full_height * 0.75)
            essential_span = essential_bottom - curr_top
            
            required_min_h = essential_span * 1.2
            if t_h < required_min_h: t_h = required_min_h
            t_h = min(t_h, src_h)
            
            t_cx = smooth_cx[i]
            bias_y = t_h * -0.05 
            t_cy = smooth_cy[i] + bias_y
            
            # Guard Rails
            head_buffer = t_h * 0.12
            safe_head_top = curr_top - head_buffer
            
            proposed_top = t_cy - t_h / 2
            if proposed_top > safe_head_top:
                t_cy = safe_head_top + t_h / 2
            
            bottom_buffer = t_h * 0.05
            safe_ess_bottom = essential_bottom + bottom_buffer
            proposed_bottom = t_cy + t_h / 2
            
            if proposed_bottom < safe_ess_bottom:
                push_down = safe_ess_bottom - proposed_bottom
                new_top = (t_cy + push_down) - t_h/2
                if new_top < curr_top:
                    t_cy += push_down
            
            if (t_cy - t_h/2) < 0: t_cy = t_h/2
            if (t_cy + t_h/2) > src_h: t_cy = src_h - t_h/2
            
            inst_target_h.append(t_h)
            inst_target_cy.append(t_cy)
            inst_target_cx.append(t_cx)
            
        # Stage 3: Smooth Targets (Target-Smoothing)
        target_win = int(clip.fps * 4.0) 
        smooth_target_h = self.smooth_data(inst_target_h, target_win, 2)
        smooth_target_cy = self.smooth_data(inst_target_cy, target_win, 2)
        smooth_target_cx = self.smooth_data(inst_target_cx, target_win, 2)
        
        crop_rects = []
        final_targets = []
        
        # Stage 4: Generate Path
        for i in range(len(smooth_cx)):
            t_h = smooth_target_h[i]
            t_cy = smooth_target_cy[i]
            t_cx = smooth_target_cx[i]
            t_w = t_h * output_aspect
            
            final_targets.append((t_cx, t_cy, t_w, t_h))
            
            linear_prog = i / len(smooth_cx)
            eased_prog = self.ease_quintic_in_out(linear_prog)
            
            if mode == "LOCKED-ON":
                accel_prog = min(1.0, linear_prog * 6.0) 
                eased_prog = self.ease_quintic_in_out(accel_prog)

            start_h = src_h; start_cx = src_w/2; start_cy = src_h/2
            
            final_h = start_h + (t_h - start_h) * eased_prog
            final_cx = start_cx + (t_cx - start_cx) * eased_prog
            final_cy = start_cy + (t_cy - start_cy) * eased_prog
            final_w = final_h * output_aspect
            
            cx1 = final_cx - final_w / 2
            cy1 = final_cy - final_h / 2
            
            if cx1 < 0: cx1 = 0
            if cy1 < 0: cy1 = 0
            if cx1 + final_w > src_w: cx1 = src_w - final_w
            if cy1 + final_h > src_h: cy1 = src_h - final_h
            
            crop_rects.append([cx1, cy1, final_w, final_h])
            
        crop_rects = np.array(crop_rects)
        
        # Stage 5: Final Steadicam Polish
        final_win = int(clip.fps * 2.0)
        final_x = self.smooth_data(crop_rects[:,0], final_win, 2)
        final_y = self.smooth_data(crop_rects[:,1], final_win, 2)
        final_w = self.smooth_data(crop_rects[:,2], final_win, 2)
        final_h = self.smooth_data(crop_rects[:,3], final_win, 2)
        
        return {
            "crop_box": (final_x, final_y, final_w, final_h),
            "target_box": final_targets,
            "motion_score": motion_score,
            "raw_rects": raw_rects,
            "raw_ratios": smooth_ratios,
            "raw_tops": smooth_tops,
            "raw_bottoms": smooth_bottoms,
            "rec_slowmo": need_slowmo,
            "mode": mode
        }

    def render_hud_monitor(self, clip, data):
        """ 渲染 HUD """
        cx, cy, cw, ch = data["crop_box"]
        targets = data["target_box"]
        raw_rects = data["raw_rects"]
        ratios = data["raw_ratios"]
        smooth_tops = data["raw_tops"]
        smooth_bottoms = data["raw_bottoms"]
        mode = data["mode"]
        
        src_w, src_h = clip.w, clip.h
        
        # [V33.2 Fix] Variable scoping
        rec_slowmo = data["rec_slowmo"]
        
        def frame_process(get_frame, t):
            raw_frame = get_frame(t)
            idx = min(int(t * clip.fps), len(cx)-1)
            
            small_viz = cv2.resize(raw_frame, (320, 180))
            _, _, s_map_small, _, _, _ = self.get_saliency_roi(small_viz)
            heatmap = cv2.applyColorMap(s_map_small, cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (src_w, src_h))
            monitor_bg = cv2.addWeighted(raw_frame, 0.85, heatmap, 0.15, 0)

            ix1, iy1 = int(cx[idx]), int(cy[idx])
            ix2, iy2 = int(cx[idx]+cw[idx]), int(cy[idx]+ch[idx])
            
            # [V33.1 Fix] 修复解包错误
            rx1 = max(0, ix1)
            ry1 = max(0, iy1)
            rx2 = min(src_w, ix2)
            ry2 = min(src_h, iy2)
            if rx2 <= rx1 or ry2 <= ry1: 
                rx1, ry1, rx2, ry2 = 0, 0, src_w, src_h
            
            crop_img = raw_frame[ry1:ry2, rx1:rx2]
            try:
                final_res_img = cv2.resize(crop_img, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            except:
                final_res_img = cv2.resize(raw_frame, (self.target_w, self.target_h))

            scale = self.target_h / src_h
            mon_w = int(src_w * scale)
            mon_img = cv2.resize(monitor_bg, (mon_w, self.target_h))
            def to_mon(x, y): return int(x * scale), int(y * scale)
            
            # Guides
            cv2.line(mon_img, (mon_w//3, 0), (mon_w//3, self.target_h), (100,100,100), 1)
            cv2.line(mon_img, (2*mon_w//3, 0), (2*mon_w//3, self.target_h), (100,100,100), 1)
            cv2.line(mon_img, (0, self.target_h//3), (mon_w, self.target_h//3), (100,100,100), 1)
            cv2.line(mon_img, (0, 2*self.target_h//3), (mon_w, 2*self.target_h//3), (100,100,100), 1)
            
            head_y = smooth_tops[idx]; _, m_head_y = to_mon(0, head_y)
            cv2.line(mon_img, (0, m_head_y), (mon_w, m_head_y), (255, 0, 255), 1)
            
            feet_y = smooth_bottoms[idx]; _, m_feet_y = to_mon(0, feet_y)
            cv2.line(mon_img, (0, m_feet_y), (mon_w, m_feet_y), (255, 0, 0), 1)
            
            draw_dashed_rect(mon_img, 0, 0, mon_w, self.target_h, (100,100,100), 1)
            
            tcx, tcy, tcw, tch = targets[idx]
            tx1, ty1 = int(tcx - tcw/2), int(tcy - tch/2)
            tx2, ty2 = int(tcx + tcw/2), int(tcy + tch/2)
            mtx1, mty1 = to_mon(tx1, ty1); mtx2, mty2 = to_mon(tx2, ty2)
            draw_dashed_rect(mon_img, mtx1, mty1, mtx2, mty2, (0, 255, 255), 1)
            
            mx1, my1 = to_mon(ix1, iy1); mx2, my2 = to_mon(ix2, iy2)
            cv2.rectangle(mon_img, (mx1, my1), (mx2, my2), (0, 255, 0), 2)
            
            curr_center = ((mx1+mx2)//2, (my1+my2)//2)
            target_center = ((mtx1+mtx2)//2, (mty1+mty2)//2)
            draw_vector_arrow(mon_img, curr_center, target_center, (0, 255, 255))
            
            sx, sy, sw, sh = raw_rects[idx]
            bx1, by1 = to_mon(sx, sy); bx2, by2 = to_mon(sx+sw, sy+sh)
            draw_bracket(mon_img, bx1, by1, bx2, by2, (0, 0, 255), thickness=1)

            ui_x, ui_y = 20, self.target_h - 60
            curr_zoom = src_h / ch[idx]
            draw_hud_label(mon_img, f"ZOOM: {curr_zoom:.2f}x", ui_x, ui_y, bg_color=(0,100,0))
            draw_hud_label(mon_img, f"CAM: {mode}", ui_x + 130, ui_y, bg_color=(100, 0, 100))
            
            spd_txt = "SPEED: 1.0x"
            rec_txt = " (REC: SLOW)" if rec_slowmo else ""
            draw_hud_label(mon_img, spd_txt + rec_txt, ui_x + 260, ui_y, bg_color=(50, 50, 50))
            
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
    engine = DirectorEngine()
    
    VIDEO_PLAYLIST = [
        r"C:\Users\Administrator\Desktop\vlog-clean\data\7.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\VID_20230422_132837.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\VID_20230422_133310.mp4",
        r"C:\Users\Administrator\Desktop\vlog-clean\data\VID_20230422_133422.mp4"
    ]
    
    print("\n" + "="*60)
    print("【AI 导演监视器 V33.2 - Bugfix Stable】")
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
            
            out_dir = "hud_v33_2_output"
            if not os.path.exists(out_dir): os.makedirs(out_dir)
            
            fname = os.path.splitext(os.path.basename(video_path))[0]
            out_name = os.path.join(out_dir, f"HUD_V33_{fname}_Stable.mp4")
            
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