#!/usr/bin/env python3
"""
批量台标提取器 — 支持 M3U 播放列表
从多个直播流中提取带 Alpha 通道的台标 (RGBA)
频道名保存，同名加序号 (_2, _3, ...)
核心算法：Welford 在线均值/方差 → 物理合成模型反解
"""
import cv2
import numpy as np
import os
import sys
import gc
import re
import argparse
import time
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
DEFAULT_NUM_FRAMES = 150
SCALE = 0.5
DEFAULT_OUTPUT_DIR = "./logo_output_batch"


# ============================================================
# M3U 解析
# ============================================================
def parse_m3u(filepath):
    """
    解析 M3U 文件 → [(channel_name, stream_url), ...]
    支持本地文件和 http(s) URL
    """
    if filepath.startswith("http://") or filepath.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(filepath, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
        # 尝试 UTF-8，失败则 GBK
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("gbk", errors="replace")
    else:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="gbk", errors="replace") as f:
                content = f.read()

    lines = content.strip().splitlines()

    channels = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("#EXTINF"):
            # 提取频道名：最后一个逗号之后的文本
            comma_idx = line.rfind(",")
            if comma_idx < 0:
                i += 1
                continue
            name = line[comma_idx + 1:].strip()

            # 提取 tvg-name / group-title 等元数据（可选）
            tvg_name_m = re.search(r'tvg-name="([^"]*)"', line)
            group_m = re.search(r'group-title="([^"]*)"', line)
            tvg_name = tvg_name_m.group(1) if tvg_name_m else name
            group = group_m.group(1) if group_m else ""

            # 下一行非注释行即为 URL
            i += 1
            while i < len(lines) and lines[i].strip().startswith("#"):
                i += 1
            if i < len(lines):
                url = lines[i].strip()
                if url and not url.startswith("#"):
                    channels.append({
                        "name": name,
                        "tvg_name": tvg_name,
                        "group": group,
                        "url": url,
                    })
        i += 1

    return channels


def build_channel_list(channels):
    """
    为同名频道添加序号。
    第一个出现的频道不加序号，后续同名频道加 _2, _3, ...
    返回 [(display_name, url, group), ...]
    """
    name_count = defaultdict(int)
    result = []

    for ch in channels:
        name = ch["name"]
        name_count[name] += 1
        seq = name_count[name]
        if seq == 1:
            display = name
        else:
            display = f"{name}_{seq}"
        result.append((display, ch["url"], ch["group"]))

    return result


# ============================================================
# 文件名安全
# ============================================================
def safe_filename(name):
    """去除文件名中的非法字符"""
    # Windows 非法字符: < > : " / \ | ? *
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # 去除首尾空格和点
    name = name.strip().strip('.')
    return name if name else "unknown"


# ============================================================
# 流式统计（Welford 在线算法）
# ============================================================
def stream_pass(url, num_frames, scale, bbox=None, verbose=True):
    """
    单遍流式处理：增量累积 mean/variance
    bbox=None → 全帧；bbox 给定 → 只处理 ROI
    返回 (mean, var, count, first_frame_uint8)
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开流: {url}")

    count = 0
    mean = None
    M2 = None
    first_frame = None

    if bbox is not None:
        x1, y1, x2, y2 = [int(v * scale) for v in bbox]

    label = "全帧" if bbox is None else f"ROI({x1},{y1})-({x2},{y2})"
    if verbose:
        print(f"    [{label}] 流式累积...")

    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if scale != 1.0:
            frame_rgb = cv2.resize(frame_rgb, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

        if bbox is not None:
            frame_rgb = frame_rgb[y1:y2, x1:x2, :]

        pixel = frame_rgb.astype(np.float64)

        if first_frame is None:
            first_frame = frame_rgb.copy()

        count += 1
        if mean is None:
            mean = pixel.copy()
            M2 = np.zeros_like(pixel)
        else:
            delta = pixel - mean
            mean += delta / count
            delta2 = pixel - mean
            M2 += delta * delta2

        if verbose and count % 50 == 0:
            print(f"      {count}/{num_frames} 帧")

    cap.release()
    gc.collect()

    var = M2 / max(count, 1)
    return mean, var, count, first_frame


# ============================================================
# 台标区域检测（通用版 — 无 CCTV1 硬编码）
# ============================================================
def find_logo_region(std_map, threshold_percentile=2):
    """
    在标准差图上定位台标区域。
    假设台标位于左上角，搜索范围 = 宽30% × 高25%。
    返回 (bbox_scaled, seg_mask)
    """
    h, w = std_map.shape

    search_x = int(w * 0.30)
    search_y = int(h * 0.25)

    std_smooth = cv2.GaussianBlur(std_map, (3, 3), 0)

    thresh = np.percentile(std_smooth, threshold_percentile)
    mask = (std_smooth <= thresh).astype(np.uint8)

    search_mask = np.zeros_like(mask)
    search_mask[:search_y, :search_x] = 1
    mask = mask * search_mask

    if mask.sum() < 50:
        thresh = np.percentile(std_smooth, 5)
        mask = (std_smooth <= thresh).astype(np.uint8)
        mask = mask * search_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)

    if num_labels <= 1:
        # 兜底：左上角固定区域
        fb_w = min(int(w * 0.25), 250)
        fb_h = min(int(h * 0.12), 110)
        return (0, 0, fb_w, fb_h), mask

    best_score = -1
    best_bbox = None

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_ = stats[i, cv2.CC_STAT_WIDTH]
        h_ = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 30:
            continue

        aspect = w_ / max(h_, 1)
        if aspect < 0.2 or aspect > 10.0:
            continue

        score = 0.0

        if x <= 3:
            score += 50
        elif x <= 8:
            score += 30
        elif x <= int(w * 0.03):
            score += 10
        else:
            continue

        if y <= 3:
            score += 30
        elif y <= int(h * 0.03):
            score += 15

        dist_score = 1.0 - (cx / max(search_x, 1) + cy / max(search_y, 1)) / 2.0
        score += dist_score * 40

        bbox_area = w_ * h_
        fill_ratio = area / max(bbox_area, 1)
        score += fill_ratio * 15

        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + w_, y + h_)

    if best_bbox is None:
        fb_w = min(int(w * 0.25), 250)
        fb_h = min(int(h * 0.12), 110)
        return (0, 0, fb_w, fb_h), mask

    # 在检测结果基础上扩展 padding
    bx1, by1, bx2, by2 = best_bbox
    pad_x = int((bx2 - bx1) * 0.25)
    pad_y = int((by2 - by1) * 0.35)
    bx1 = max(0, bx1 - pad_x)
    by1 = max(0, by1 - pad_y)
    bx2 = min(w - 1, bx2 + pad_x)
    by2 = min(h - 1, by2 + pad_y)

    return (bx1, by1, bx2, by2), mask


# ============================================================
# Alpha 反解（通用版）
# ============================================================
def solve_alpha_from_stats(mean_I, var_I, mean_full=None):
    """
    从均值/方差统计反解台标 alpha 和颜色。
    物理模型: I = alpha * L + (1 - alpha) * B
    两级 alpha: 文字=1.0, 底板=固定值 0.80
    """
    h, w = mean_I.shape[:2]
    eps = 1e-7

    mean_I_norm = mean_I.astype(np.float64) / 255.0
    var_I_norm = var_I.astype(np.float64) / (255.0 * 255.0)
    var_I_gray = var_I_norm.mean(axis=2)
    std_I = np.sqrt(np.clip(var_I_gray, 0, None))
    mean_gray = mean_I_norm.mean(axis=2)

    # 背景方差估计
    var_B_est = np.percentile(var_I_gray, 95)
    var_B_est = max(var_B_est, 0.005)
    std_B = np.sqrt(var_B_est)
    alpha_var = np.clip(1.0 - std_I / (std_B + eps), 0, 1)

    # 暗度信号
    if mean_full is not None:
        mean_full_norm = mean_full.astype(np.float64) / 255.0
        mean_full_gray = mean_full_norm.mean(axis=2)
        mean_full_gray_uint8 = (np.clip(mean_full_gray, 0, 1) * 255).astype(np.uint8)
        local_bg_full = cv2.medianBlur(mean_full_gray_uint8, 101).astype(np.float64) / 255.0
        dark_score_full = np.clip(local_bg_full - mean_full_gray, 0, 1)
        dark_score = dark_score_full[:h, :w]
    else:
        mean_gray_uint8 = (mean_gray * 255).astype(np.uint8)
        local_bg = cv2.medianBlur(mean_gray_uint8, 15).astype(np.float64) / 255.0
        dark_score = np.clip(local_bg - mean_gray, 0, 1)

    # 文字种子
    text_seed = alpha_var > 0.60
    text_count = text_seed.sum()

    if text_count < 30:
        text_seed = alpha_var > 0.40
        text_count = text_seed.sum()

    text_clean = cv2.morphologyEx(text_seed.astype(np.uint8),
                                  cv2.MORPH_OPEN,
                                  np.ones((2, 2), np.uint8)).astype(bool)

    # 只保留左上角
    search_h = int(h * 0.7)
    search_w = int(w * 0.8)
    corner_mask = np.zeros_like(text_clean)
    corner_mask[:search_h, :search_w] = True
    text_clean = text_clean & corner_mask

    # 文字 bbox 约束 + 比例扩展
    text_ys, text_xs = np.where(text_clean)
    if len(text_ys) > 0:
        tx1, tx2 = text_xs.min(), text_xs.max()
        ty1, ty2 = text_ys.min(), text_ys.max()
        text_h = max(ty2 - ty1, 10)
        text_w = max(tx2 - tx1, 10)
        pad_x = int(text_w * 0.30)
        pad_y = int(text_h * 0.40)
        bbox_x1 = max(0, tx1 - pad_x)
        bbox_x2 = min(w - 1, tx2 + pad_x)
        bbox_y1 = max(0, ty1 - pad_y)
        bbox_y2 = min(h - 1, ty2 + pad_y)

        bbox_constraint = np.zeros_like(text_clean)
        bbox_constraint[bbox_y1:bbox_y2 + 1, bbox_x1:bbox_x2 + 1] = True
    else:
        bbox_constraint = np.ones_like(text_clean, dtype=bool)

    # 闭运算扩展到底板
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 6))
    closed = cv2.morphologyEx(text_clean.astype(np.uint8),
                              cv2.MORPH_CLOSE, kernel_close).astype(bool)

    logo_mask = closed & bbox_constraint

    logo_mask = cv2.morphologyEx(logo_mask.astype(np.uint8),
                                 cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8)).astype(bool)

    logo_pixel_count = logo_mask.sum()

    if logo_pixel_count == 0:
        fixed_h = int(h * 0.5)
        fixed_w = int(w * 0.6)
        logo_mask[:fixed_h, :fixed_w] = True
        logo_pixel_count = logo_mask.sum()

    is_logo = logo_mask

    # 两级 alpha
    alpha_from_var = np.clip(alpha_var ** 1.5 * 2.0, 0, 1)
    is_text = is_logo & text_clean
    alpha = np.where(is_text, 1.0, alpha_from_var)
    is_plate = is_logo & ~is_text
    alpha = np.where(is_plate, np.maximum(alpha_from_var, 0.80), alpha)
    alpha = np.where(is_logo, alpha, 0.0)
    alpha = np.clip(alpha, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.where(is_logo, alpha, 0.0)
    alpha = np.clip(alpha, 0, 1)

    # 精化颜色
    L = np.zeros_like(mean_I_norm, dtype=np.float64)
    L[:] = mean_I_norm

    if logo_pixel_count > 0:
        B_local = np.zeros_like(mean_I_norm, dtype=np.float64)
        B_local[:] = mean_I_norm

        logo_ys, logo_xs = np.where(is_logo)
        bbox_x1, bbox_x2 = logo_xs.min(), logo_xs.max()
        bbox_y1, bbox_y2 = logo_ys.min(), logo_ys.max()

        margin = 5
        sample_xs, sample_ys, sample_colors = [], [], []

        for sy_line in [min(bbox_y2 + margin, h - 1), max(bbox_y1 - margin, 0)]:
            for sx in range(max(0, bbox_x1 - margin), min(bbox_x2 + margin + 1, w)):
                if not is_logo[sy_line, sx]:
                    sample_xs.append(sx)
                    sample_ys.append(sy_line)
                    sample_colors.append(mean_I_norm[sy_line, sx])

        for sx_line in [min(bbox_x2 + margin, w - 1), max(bbox_x1 - margin, 0)]:
            for sy in range(max(0, bbox_y1 - margin), min(bbox_y2 + margin + 1, h)):
                if not is_logo[sy, sx_line]:
                    sample_xs.append(sx_line)
                    sample_ys.append(sy)
                    sample_colors.append(mean_I_norm[sy, sx_line])

        n_samples = len(sample_xs)

        if n_samples >= 4:
            MAX_SAMPLES = 200
            if n_samples > MAX_SAMPLES:
                indices = np.random.RandomState(42).choice(
                    n_samples, MAX_SAMPLES, replace=False)
                sample_pts = np.array(
                    [[sample_xs[i], sample_ys[i]] for i in indices],
                    dtype=np.float64)
                sample_vals = np.array(
                    [sample_colors[i] for i in indices], dtype=np.float64)
            else:
                sample_pts = np.array(
                    list(zip(sample_xs, sample_ys)), dtype=np.float64)
                sample_vals = np.array(sample_colors, dtype=np.float64)

            center = sample_pts.mean(axis=0)
            max_dist = np.max(
                np.linalg.norm(sample_pts - center, axis=1)) + eps
            sample_pts_norm = (sample_pts - center) / max_dist
            logo_pts = np.column_stack([logo_xs, logo_ys]).astype(np.float64)
            logo_pts_norm = (logo_pts - center) / max_dist

            BATCH = 2000
            for start in range(0, len(logo_pts), BATCH):
                end = min(start + BATCH, len(logo_pts))
                dists = np.linalg.norm(
                    logo_pts_norm[start:end, np.newaxis, :]
                    - sample_pts_norm[np.newaxis, :, :],
                    axis=2
                )
                weights = 1.0 / (dists ** 2 + 1e-8)
                w_sum = weights.sum(axis=1, keepdims=True)
                for c in range(3):
                    B_local[logo_ys[start:end], logo_xs[start:end], c] = \
                        (weights @ sample_vals[:, c]) / w_sum.ravel()
        else:
            non_logo_mask = ~is_logo
            if non_logo_mask.sum() > 0:
                for c in range(3):
                    B_local[is_logo, c] = np.median(
                        mean_I_norm[non_logo_mask, c])

        a = alpha.astype(np.float64)
        mask = is_logo & (a > 0.01)
        for c in range(3):
            m = mean_I_norm[:, :, c]
            b_local = B_local[:, :, c]
            L[:, :, c] = np.where(mask,
                                  (m - (1 - a) * b_local) / (a + eps),
                                  m)
            L[:, :, c] = np.clip(L[:, :, c], 0, 1)

        is_solid = (var_I_gray < var_B_est * 0.03) & is_logo
        for c in range(3):
            L[:, :, c] = np.where(is_solid, mean_I_norm[:, :, c], L[:, :, c])

    # 最终平滑
    alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.where(is_logo, alpha_smooth, 0.0).astype(np.float32)
    alpha = np.clip(alpha, 0, 1)
    L = np.clip(L, 0, 1).astype(np.float32)

    return alpha, L, mean_I_norm.astype(np.float32)


# ============================================================
# 单频道处理
# ============================================================
def process_channel(url, channel_name, output_dir, num_frames, verbose=False):
    """
    处理单个频道，提取台标并保存。
    返回结果 dict。
    """
    result = {
        "name": channel_name,
        "url": url,
        "status": "unknown",
        "output": "",
        "mask_pct": 0,
        "frames": 0,
        "mae": -1,
    }

    try:
        # ---- Pass 1: 全帧统计 ----
        mean_full, var_full, count1, first_frame = stream_pass(
            url, num_frames, SCALE, verbose=verbose)

        if count1 < 10:
            raise RuntimeError(f"帧数不足: {count1}")

        std_map = np.sqrt(np.clip(var_full, 0, None)).mean(axis=2)

        # 保存第一帧（用于调试）
        frame_dir = os.path.join(output_dir, safe_filename(channel_name))
        os.makedirs(frame_dir, exist_ok=True)
        cv2.imwrite(os.path.join(frame_dir, "first_frame.png"),
                    cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

        # 定位 logo（自动检测，仅用于日志和兜底）
        bbox_detected, seg_mask = find_logo_region(std_map)
        dx1, dy1, dx2, dy2 = bbox_detected

        if verbose:
            print(f"    自动检测(缩放): ({dx1},{dy1})-({dx2},{dy2}), "
                  f"大小={dx2-dx1}x{dy2-dy1}")

        # 根据实际分辨率自适应缩放固定 ROI
        # 参考基准：1920x1080 → ROI (0,0)-(460,200)
        # 标清频道（如 720x576）会按比例缩小，避免裁剪区域过大
        fh, fw = first_frame.shape[:2]  # 缩放后的帧尺寸
        actual_w = int(fw / SCALE)
        actual_h = int(fh / SCALE)
        REF_W, REF_H = 1920, 1080
        REF_ROI_W, REF_ROI_H = 460, 200
        scale_x = actual_w / REF_W
        scale_y = actual_h / REF_H
        roi_w = max(40, int(REF_ROI_W * scale_x))  # 最小 40 防止过小
        roi_h = max(20, int(REF_ROI_H * scale_y))  # 最小 20 防止过小
        FIXED_ROI_ORIG = (0, 0, roi_w, roi_h)
        FIXED_ROI_SCALED = (0, 0, int(roi_w * SCALE), int(roi_h * SCALE))

        if verbose:
            print(f"    分辨率: {actual_w}x{actual_h}, "
                  f"固定ROI(原始): {FIXED_ROI_ORIG}")

        # 如果自动检测区域比固定 ROI 大但不超过上限，使用自动检测（罕见情况）
        # 上限约束：防止隔行扫描等异常流导致检测区域过大
        det_w, det_h = dx2 - dx1, dy2 - dy1
        fix_w = int(roi_w * SCALE)
        fix_h = int(roi_h * SCALE)
        MAX_AUTO_W = int(fix_w * 1.6)
        MAX_AUTO_H = int(fix_h * 1.6)
        if (det_w > fix_w and det_h > fix_h
                and det_w <= MAX_AUTO_W and det_h <= MAX_AUTO_H):
            bbox_scaled = bbox_detected
            roi_source = "auto"
        else:
            if det_w > MAX_AUTO_W or det_h > MAX_AUTO_H:
                if verbose:
                    print(f"    自动检测区域过大 ({det_w}x{det_h})，"
                          f"回退到固定 ROI")
            bbox_scaled = FIXED_ROI_SCALED
            roi_source = "fixed"

        x1s, y1s, x2s, y2s = bbox_scaled

        if verbose:
            print(f"    使用 ROI({roi_source}): ({x1s},{y1s})-({x2s},{y2s})")

        # 画检测框（在缩放后的第一帧上）
        vis = first_frame.copy()
        cv2.rectangle(vis, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
        cv2.rectangle(vis, (x1s, y1s), (x2s, y2s), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(frame_dir, "detected.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        del var_full, std_map, seg_mask, vis
        gc.collect()

        # ---- Pass 2: ROI 统计 ----
        # bbox_scaled 是缩放后坐标，stream_pass 内部会再乘 scale
        # 所以这里传"原始坐标" = bbox_scaled / scale
        inv_scale = 1.0 / SCALE
        bbox_orig = (int(x1s * inv_scale), int(y1s * inv_scale),
                     int(x2s * inv_scale), int(y2s * inv_scale))
        mean_roi, var_roi, count2, first_roi = stream_pass(
            url, num_frames, SCALE, bbox=bbox_orig, verbose=verbose)

        if count2 < 10:
            raise RuntimeError(f"ROI 帧数不足: {count2}")

        # ---- 反解 alpha ----
        alpha, logo_rgb, bg_est = solve_alpha_from_stats(
            mean_roi, var_roi, mean_full=mean_full)

        del mean_full
        gc.collect()

        # 放大回原始尺寸
        h_s, w_s = alpha.shape[:2]
        if SCALE != 1.0:
            orig_h = int(h_s / SCALE)
            orig_w = int(w_s / SCALE)
            alpha = cv2.resize(alpha, (orig_w, orig_h),
                               interpolation=cv2.INTER_CUBIC)
            logo_rgb = cv2.resize(logo_rgb, (orig_w, orig_h),
                                  interpolation=cv2.INTER_CUBIC)
            bg_est = cv2.resize(bg_est, (orig_w, orig_h),
                                interpolation=cv2.INTER_CUBIC)

        h_out, w_out = alpha.shape[:2]

        # ---- 保存结果 ----
        alpha_uint8 = (alpha * 255).astype(np.uint8)

        # RGBA
        logo_bgr = cv2.cvtColor((logo_rgb * 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR)
        rgba = np.zeros((h_out, w_out, 4), dtype=np.uint8)
        rgba[:, :, :3] = logo_bgr
        rgba[:, :, 3] = alpha_uint8

        rgba_path = os.path.join(output_dir,
                                 f"{safe_filename(channel_name)}.png")
        cv2.imwrite(rgba_path, rgba)

        # 黑底合成
        alpha_3c = np.stack([alpha.astype(np.float32)] * 3, axis=2)
        comp_black = alpha_3c * logo_rgb
        comp_black_bgr = cv2.cvtColor(
            (np.clip(comp_black, 0, 1) * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(frame_dir, "composite_black.png"),
                    comp_black_bgr)

        # 重建验证
        if first_roi is not None:
            first_roi_f = first_roi.astype(np.float32) / 255.0
            if SCALE != 1.0:
                first_roi_f = cv2.resize(first_roi_f, (w_out, h_out),
                                         interpolation=cv2.INTER_CUBIC)
            recon = alpha_3c * logo_rgb + (1 - alpha_3c) * bg_est
            mae = np.abs(first_roi_f - recon).mean()
            cmp = np.hstack([first_roi_f, recon,
                             np.clip(np.abs(first_roi_f - recon) * 10, 0, 1)])
            cv2.imwrite(os.path.join(frame_dir, "recon_verify.png"),
                        cv2.cvtColor((cmp * 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))
        else:
            mae = -1

        mask_pct = (alpha > 0.01).sum() / max(h_out * w_out, 1)

        result.update({
            "status": "ok",
            "output": rgba_path,
            "mask_pct": mask_pct,
            "frames": count1,
            "mae": mae,
            "size": f"{w_out}x{h_out}",
            "roi_source": roi_source,
        })

    except RuntimeError as e:
        result["status"] = f"error: {e}"
    except Exception as e:
        result["status"] = f"error: {e}"
    finally:
        gc.collect()

    return result


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="批量台标提取器 — 支持 M3U 播放列表")
    parser.add_argument("m3u", help="M3U 文件路径或 URL")
    parser.add_argument("--frames", type=int, default=DEFAULT_NUM_FRAMES,
                        help=f"每流处理帧数 (默认 {DEFAULT_NUM_FRAMES})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help=f"输出目录 (默认 {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--filter", default=None,
                        help="频道名过滤 (子串匹配，如 'CCTV')")
    parser.add_argument("--limit", type=int, default=0,
                        help="最多处理 N 个频道 (0=全部)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="跳过已有输出的频道 (默认开启)")
    parser.add_argument("--no-skip", action="store_true", default=False,
                        help="不跳过，强制重新处理")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="显示详细日志")
    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip

    print("=" * 60)
    print("  批量台标提取器 v1.1 (固定ROI + 坐标修正)")
    print(f"  播放列表: {args.m3u}")
    print(f"  帧数/流: {args.frames}, 缩放: {SCALE}")
    print(f"  输出目录: {args.output}")
    print("=" * 60)

    # 解析 M3U
    print("\n[1] 解析播放列表...")
    channels_raw = parse_m3u(args.m3u)
    print(f"    解析到 {len(channels_raw)} 个流条目")

    # 构建带序号的频道列表
    channel_list = build_channel_list(channels_raw)
    unique_names = set(ch[0].split("_")[0] if ch[0].split("_")[0] in
                       [c["name"] for c in channels_raw] else ch[0]
                       for ch in channel_list)

    # 统计同名
    name_groups = defaultdict(list)
    for ch in channels_raw:
        name_groups[ch["name"]].append(ch)
    dup_count = sum(1 for v in name_groups.values() if len(v) > 1)
    print(f"    独立频道: {len(name_groups)}, "
          f"有同名的: {dup_count}, "
          f"总条目: {len(channel_list)}")

    # 过滤
    if args.filter:
        channel_list = [(n, u, g) for n, u, g in channel_list
                        if args.filter in n]
        print(f"    过滤 '{args.filter}': {len(channel_list)} 个")

    if args.limit > 0:
        channel_list = channel_list[:args.limit]
        print(f"    限制前 {args.limit} 个")

    total = len(channel_list)
    if total == 0:
        print("    无频道可处理，退出。")
        return

    print(f"\n[2] 开始处理 {total} 个频道...\n")

    os.makedirs(args.output, exist_ok=True)

    results = []
    ok_count = 0
    fail_count = 0
    skip_count = 0
    t_start = time.time()

    for idx, (display_name, url, group) in enumerate(channel_list):
        prefix = f"[{idx + 1}/{total}]"
        safe_name = safe_filename(display_name)
        out_path = os.path.join(args.output, f"{safe_name}.png")

        # 断点续传
        if skip_existing and os.path.exists(out_path):
            print(f"{prefix} 跳过 {display_name} (已存在)")
            skip_count += 1
            results.append({
                "name": display_name,
                "url": url,
                "group": group,
                "status": "skipped",
                "output": out_path,
            })
            continue

        print(f"{prefix} {display_name} ({group})")
        print(f"    URL: {url[:80]}...")

        result = process_channel(url, display_name, args.output,
                                 args.frames, verbose=args.verbose)
        result["group"] = group
        results.append(result)

        elapsed = time.time() - t_start
        eta = elapsed / max(idx + 1 - skip_count, 1) * (
                total - idx - 1) if (idx + 1 - skip_count) > 0 else 0

        if result["status"] == "ok":
            ok_count += 1
            roi = result.get("roi_source", "?")
            print(f"    ✓ mask={result['mask_pct']:.1%}, "
                  f"MAE={result.get('mae', -1):.3f}, "
                  f"尺寸={result.get('size', '?')}, "
                  f"ROI={roi}, "
                  f"耗时 {elapsed:.0f}s (ETA {eta:.0f}s)")
        else:
            fail_count += 1
            print(f"    ✗ {result['status']}")

        # 写进度日志
        with open(os.path.join(args.output, "progress.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"{idx + 1}/{total} {display_name}: {result['status']}\n")

        sys.stdout.flush()

    # ---- 汇总 ----
    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  完成！耗时 {total_time:.0f}s")
    print(f"  成功: {ok_count}, 失败: {fail_count}, 跳过: {skip_count}")
    print(f"  输出目录: {args.output}")
    print(f"{'=' * 60}")

    # 写汇总
    summary_path = os.path.join(args.output, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"批量台标提取 — 汇总报告\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"播放列表: {args.m3u}\n")
        f.write(f"帧数/流: {args.frames}\n")
        f.write(f"总频道: {total}, 成功: {ok_count}, "
                f"失败: {fail_count}, 跳过: {skip_count}\n")
        f.write(f"耗时: {total_time:.0f}s\n\n")

        f.write("--- 成功 ---\n")
        for r in results:
            if r["status"] == "ok":
                roi = r.get("roi_source", "?")
                f.write(f"  {r['name']:30s}  "
                        f"mask={r.get('mask_pct', 0):.1%}  "
                        f"MAE={r.get('mae', -1):.3f}  "
                        f"尺寸={r.get('size', '?')}  "
                        f"ROI={roi}\n")

        f.write("\n--- 失败 ---\n")
        for r in results:
            if r["status"] not in ("ok", "skipped"):
                f.write(f"  {r['name']:30s}  {r['status']}\n")
                f.write(f"    URL: {r['url']}\n")

        f.write("\n--- 跳过 ---\n")
        for r in results:
            if r["status"] == "skipped":
                f.write(f"  {r['name']:30s}  {r['output']}\n")

    print(f"  汇总: {summary_path}")


if __name__ == "__main__":
    main()
