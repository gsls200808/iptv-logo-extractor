#!/usr/bin/env python3
"""
从直播流提取带 Alpha 通道的台标 (极低内存版)
核心思路：全部用增量统计，不存储任何帧数组
物理模型: I = alpha*L + (1-alpha)*B
"""
import cv2
import numpy as np
import os
import gc

STREAM_URL = "http://hwottcdn.ln.chinamobile.com/PLTV/11/224/3221226190/index.m3u8"
NUM_FRAMES = 300
OUTPUT_DIR = "./logo_output"
SCALE = 0.5  # 缩小到一半处理，大幅降低内存
os.makedirs(OUTPUT_DIR, exist_ok=True)


def stream_pass(url, num_frames, scale, bbox=None):
    """
    单遍流式处理：增量累积 mean/variance
    如果 bbox=None，处理全帧（用于定位 logo）
    如果 bbox 给定，只处理 ROI
    返回 (mean, var, count, first_frame_uint8)
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开流: {url}")

    # Welford 在线算法
    count = 0
    mean = None
    M2 = None  # sum of (x - mean)^2
    first_frame = None

    if bbox is not None:
        x1, y1, x2, y2 = [int(v * scale) for v in bbox]

    label = "全帧" if bbox is None else f"ROI({x1},{y1})-({x2},{y2})"
    print(f"  [{label}] 流式累积统计...")

    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if scale != 1.0:
            frame_rgb = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

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

        if count % 100 == 0:
            print(f"    {count}/{num_frames} 帧")

    cap.release()
    gc.collect()

    var = M2 / max(count, 1)
    return mean, var, count, first_frame


def find_logo_region(std_map, threshold_percentile=2):
    """
    标准差图上找台标区域（专为 CCTV 台标优化）。
    CCTV 台标特征：
    - 固定在左上角，紧贴左边缘和顶边缘
    - 有暗色圆角矩形背景（极低方差）
    - 白色半透明文字（中等方差）
    策略：用极低阈值找暗色背景 → 形态学扩展 → 强制贴近左边缘
    """
    h, w = std_map.shape

    # 严格限制搜索区域到左上角（CCTV 台标紧贴角落）
    search_x = int(w * 0.25)
    search_y = int(h * 0.20)

    # 轻微模糊去噪
    std_smooth = cv2.GaussianBlur(std_map, (3, 3), 0)

    thresh = np.percentile(std_smooth, threshold_percentile)
    mask = (std_smooth <= thresh).astype(np.uint8)

    # 只保留左上角搜索区
    search_mask = np.zeros_like(mask)
    search_mask[:search_y, :search_x] = 1
    mask = mask * search_mask

    if mask.sum() < 50:
        # 阈值太严，放宽到 5%
        thresh = np.percentile(std_smooth, 5)
        mask = (std_smooth <= thresh).astype(np.uint8)
        mask = mask * search_mask

    # 闭运算填补台标背景内部空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        # 兜底：固定截取左上角 460x200（原始分辨率），缩放后 230x100
        return (0, 0, 230, 100), mask

    best_score = -1
    best_bbox = None

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w_ = stats[i, cv2.CC_STAT_WIDTH]
        h_ = stats[i, cv2.CC_STAT_HEIGHT]

        if area < 30:  # 降低面积阈值
            continue

        # 宽高比（放宽范围）
        aspect = w_ / max(h_, 1)
        if aspect < 0.3 or aspect > 8.0:
            continue

        score = 0.0

        # 必须非常贴近左边缘（台标紧贴左边）
        if x <= 3:
            score += 50
        elif x <= 8:
            score += 30
        elif x <= int(w * 0.03):
            score += 10
        else:
            continue  # 不贴近左边的直接跳过

        # 贴近顶边缘
        if y <= 3:
            score += 30
        elif y <= int(h * 0.03):
            score += 15

        # 越靠近左上角越好
        dist_score = 1.0 - (cx / search_x + cy / search_y) / 2.0
        score += dist_score * 40

        # 紧凑度
        bbox_area = w_ * h_
        fill_ratio = area / max(bbox_area, 1)
        score += fill_ratio * 15

        if score > best_score:
            best_score = score
            best_bbox = (x, y, x + w_, y + h_)

    if best_bbox is None:
        # 兜底：固定截取左上角 460x200（原始分辨率），缩放后 230x100
        return (0, 0, 230, 100), mask

    # 强制固定 ROI：左上角 460x200（原始分辨率），缩放后 230x100
    # CCTV 台标紧贴左上角，固定 ROI 确保完整捕获
    x1, y1 = 0, 0
    x2, y2 = 230, 100  # 230x100 at 0.5x = 460x200 at original

    return (x1, y1, x2, y2), mask


def solve_alpha_from_stats(mean_I, var_I, mean_full=None):
    """
    v20：全帧暗度信号 + 收紧 mask + dark_score 仅用于 mask 内 alpha 精化
    物理模型: I = alpha * L + (1 - alpha) * B

    v19 问题：dark_score 在 ROI(230×100) 上计算，medianBlur 核 31 比 ROI 高度还大，
              局部背景被台标自身污染 → dark_score 不可靠 → mask 膨胀到 49.6%
    v20 改进：
      1. dark_score 在全帧 mean_full 上计算（medianBlur 有充足背景上下文）
      2. 闭运算核从 (10,7) 收紧到 (7,5)，减少 mask 膨胀
      3. dark_score 仅用于 mask 内部的 alpha 精化，不参与检测
    """
    h, w = mean_I.shape[:2]
    eps = 1e-7

    # 1. 归一化到 [0, 1] 空间
    mean_I_norm = mean_I.astype(np.float64) / 255.0
    var_I_norm = var_I.astype(np.float64) / (255.0 * 255.0)
    var_I_gray = var_I_norm.mean(axis=2)
    std_I = np.sqrt(np.clip(var_I_gray, 0, None))
    mean_gray = mean_I_norm.mean(axis=2)  # 灰度均值图

    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  var_I 百分位 P{p}: {np.percentile(var_I_gray, p):.6f}")
    print(f"  mean_gray 范围: [{mean_gray.min():.4f}, {mean_gray.max():.4f}], "
          f"中位数: {np.median(mean_gray):.4f}")

    # ================================================================
    # 2. 估计背景方差 & 计算 alpha_var
    # ================================================================
    var_B_est = np.percentile(var_I_gray, 95)
    var_B_est = max(var_B_est, 0.005)
    std_B = np.sqrt(var_B_est)
    alpha_var = np.clip(1.0 - std_I / (std_B + eps), 0, 1)

    print(f"  var_B (P95): {var_B_est:.6f}, std_B: {std_B:.6f}")
    print(f"  alpha_var 范围: [{alpha_var.min():.4f}, {alpha_var.max():.4f}], "
          f"中位数: {np.median(alpha_var):.4f}")

    # ================================================================
    # 2b. 暗度信号：在全帧上计算（关键改进！）
    # ================================================================
    # v19 问题：在 ROI(230×100) 上 medianBlur(31)，核比 ROI 高度还大
    #          → 局部背景被台标暗色污染 → dark_score 不可靠
    # v20 解决：在全帧 mean_full 上计算，medianBlur 有充足背景上下文
    if mean_full is not None:
        print(f"  ★ 使用全帧 mean 计算 dark_score（核 101，背景上下文充足）")
        mean_full_norm = mean_full.astype(np.float64) / 255.0
        mean_full_gray = mean_full_norm.mean(axis=2)
        mean_full_gray_uint8 = (np.clip(mean_full_gray, 0, 1) * 255).astype(np.uint8)
        # v21：核从 31 增大到 101
        # 台标约 180×80 像素，核 31 的邻域仍被台标暗色污染
        # 核 101 覆盖约 1.9% 全帧面积，中值由真正背景主导
        local_bg_full = cv2.medianBlur(mean_full_gray_uint8, 101).astype(np.float64) / 255.0
        dark_score_full = np.clip(local_bg_full - mean_full_gray, 0, 1)
        # 裁剪到 ROI 尺寸
        dark_score = dark_score_full[:h, :w]
    else:
        print(f"  ⚠ 无全帧 mean，在 ROI 上计算 dark_score（效果较差）")
        mean_gray_uint8 = (mean_gray * 255).astype(np.uint8)
        local_bg = cv2.medianBlur(mean_gray_uint8, 15).astype(np.float64) / 255.0
        dark_score = np.clip(local_bg - mean_gray, 0, 1)

    for p in [50, 75, 90, 95, 99]:
        print(f"  dark_score 百分位 P{p}: {np.percentile(dark_score, p):.4f}")

    # ================================================================
    # 3. 检测台标区域（文字种子 + 适度闭运算扩展到底板）
    # ================================================================
    # v23 策略：放弃 dark_score 做检测（流内容变化时阈值不可靠）
    # 改用两级 alpha：文字=1.0，底板=固定值 0.80
    # 闭运算从文字种子扩展到底板区域

    # 文字种子：高 alpha_var（v24：阈值提高到 0.60）
    text_seed = alpha_var > 0.60
    text_count = text_seed.sum()
    print(f"  文字种子 (alpha_var>0.60): {text_count} ({text_count/(h*w):.1%})")

    if text_count < 30:
        text_seed = alpha_var > 0.40
        text_count = text_seed.sum()
        print(f"  放宽文字种子 (alpha_var>0.40): {text_count}")

    # 轻开运算去噪
    text_clean = cv2.morphologyEx(text_seed.astype(np.uint8),
                                  cv2.MORPH_OPEN,
                                  np.ones((2, 2), np.uint8)).astype(bool)

    # 只保留左上角
    search_h = int(h * 0.7)
    search_w = int(w * 0.8)
    corner_mask = np.zeros_like(text_clean)
    corner_mask[:search_h, :search_w] = True
    text_clean = text_clean & corner_mask

    # 文字 bbox 约束
    text_ys, text_xs = np.where(text_clean)
    if len(text_ys) > 0:
        tx1, tx2 = text_xs.min(), text_xs.max()
        ty1, ty2 = text_ys.min(), text_ys.max()
        # v24：bbox 按文字尺寸比例扩展（25%），避免固定像素在不同流中不适应
        text_h = max(ty2 - ty1, 10)
        text_w = max(tx2 - tx1, 10)
        pad_x = int(text_w * 0.30)  # 水平扩展 30%
        pad_y = int(text_h * 0.40)  # 垂直扩展 40%（底板在文字上下都有延伸）
        bbox_x1 = max(0, tx1 - pad_x)
        bbox_x2 = min(w - 1, tx2 + pad_x)
        bbox_y1 = max(0, ty1 - pad_y)
        bbox_y2 = min(h - 1, ty2 + pad_y)

        bbox_constraint = np.zeros_like(text_clean)
        bbox_constraint[bbox_y1:bbox_y2+1, bbox_x1:bbox_x2+1] = True
        print(f"  文字 bbox 约束(扩展): y=[{bbox_y1},{bbox_y2}], x=[{bbox_x1},{bbox_x2}]")
    else:
        bbox_constraint = np.ones_like(text_clean, dtype=bool)

    # v24：闭运算核 (9×6)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 6))
    closed = cv2.morphologyEx(text_clean.astype(np.uint8),
                              cv2.MORPH_CLOSE, kernel_close).astype(bool)

    logo_mask = closed & bbox_constraint

    # 轻开运算去噪
    logo_mask = cv2.morphologyEx(logo_mask.astype(np.uint8),
                                 cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8)).astype(bool)

    # 添加"综合"固定矩形
    zonghe_rect = np.zeros_like(logo_mask)
    zonghe_rect[70:93, 50:170] = True
    logo_mask = logo_mask | zonghe_rect
    print(f"  添加'综合'固定矩形: y=[70,93), x=[50,170)")

    logo_pixel_count = logo_mask.sum()
    print(f"  台标总像素: {logo_pixel_count} ({logo_pixel_count/(h*w):.1%})")

    if logo_pixel_count == 0:
        fixed_h = int(h * 0.5)
        fixed_w = int(w * 0.6)
        logo_mask[:fixed_h, :fixed_w] = True
        logo_pixel_count = logo_mask.sum()
        print(f"  ★ 检测失败，使用固定区域: {logo_pixel_count} 像素")

    is_logo = logo_mask

    # ================================================================
    # 4. 计算 Alpha（v23：两级 alpha — 文字=1.0，底板=0.80）
    # ================================================================
    # v23 策略：放弃 dark_score 做 alpha（流内容变化时不可靠）
    # 文字像素：alpha = 1.0（完全不透明）
    # 底板像素：alpha = 0.80（半透明，让底板可见但不遮挡背景）

    # 基础 alpha：方差映射（用于边缘平滑过渡）
    alpha_from_var = np.clip(alpha_var ** 1.5 * 2.0, 0, 1)

    # 文字像素强制 alpha=1.0
    is_text = is_logo & text_clean
    alpha = np.where(is_text, 1.0, alpha_from_var)

    # 底板像素（在 logo mask 内但非文字）：使用方差 alpha 和固定值 0.80 的最大值
    is_plate = is_logo & ~is_text
    alpha = np.where(is_plate, np.maximum(alpha_from_var, 0.80), alpha)

    # 非 logo 区域归零
    alpha = np.where(is_logo, alpha, 0.0)
    alpha = np.clip(alpha, 0, 1)

    # 高斯模糊平滑边缘
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.where(is_logo, alpha, 0.0)
    alpha = np.clip(alpha, 0, 1)

    print(f"  alpha 范围: [{alpha[is_logo].min():.4f}, {alpha[is_logo].max():.4f}]")
    print(f"  alpha 中位数(台标): {np.median(alpha[is_logo]):.4f}")
    print(f"  alpha>0.9 占比: {(alpha[is_logo] > 0.9).sum() / max(logo_pixel_count, 1):.1%}")
    print(f"  alpha<0.1 占比: {(alpha[is_logo] < 0.1).sum() / max(logo_pixel_count, 1):.1%}")

    # ================================================================
    # 5. 精化 Logo 颜色 (L)
    # ================================================================
    L = np.zeros_like(mean_I_norm, dtype=np.float64)
    L[:] = mean_I_norm

    if logo_pixel_count > 0:
        # 5a. 边界 IDW 插值估计局部背景颜色
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
                    sample_xs.append(sx); sample_ys.append(sy_line)
                    sample_colors.append(mean_I_norm[sy_line, sx])

        for sx_line in [min(bbox_x2 + margin, w - 1), max(bbox_x1 - margin, 0)]:
            for sy in range(max(0, bbox_y1 - margin), min(bbox_y2 + margin + 1, h)):
                if not is_logo[sy, sx_line]:
                    sample_xs.append(sx_line); sample_ys.append(sy)
                    sample_colors.append(mean_I_norm[sy, sx_line])

        n_samples = len(sample_xs)
        print(f"  边界采样点: {n_samples}")

        if n_samples >= 4:
            MAX_SAMPLES = 200
            if n_samples > MAX_SAMPLES:
                indices = np.random.RandomState(42).choice(n_samples, MAX_SAMPLES, replace=False)
                sample_pts = np.array([[sample_xs[i], sample_ys[i]] for i in indices], dtype=np.float64)
                sample_vals = np.array([sample_colors[i] for i in indices], dtype=np.float64)
            else:
                sample_pts = np.array(list(zip(sample_xs, sample_ys)), dtype=np.float64)
                sample_vals = np.array(sample_colors, dtype=np.float64)

            center = sample_pts.mean(axis=0)
            max_dist = np.max(np.linalg.norm(sample_pts - center, axis=1)) + eps
            sample_pts_norm = (sample_pts - center) / max_dist
            logo_pts = np.column_stack([logo_xs, logo_ys]).astype(np.float64)
            logo_pts_norm = (logo_pts - center) / max_dist

            BATCH = 2000
            for start in range(0, len(logo_pts), BATCH):
                end = min(start + BATCH, len(logo_pts))
                dists = np.linalg.norm(
                    logo_pts_norm[start:end, np.newaxis, :] - sample_pts_norm[np.newaxis, :, :],
                    axis=2
                )
                weights = 1.0 / (dists ** 2 + 1e-8)
                w_sum = weights.sum(axis=1, keepdims=True)
                for c in range(3):
                    B_local[logo_ys[start:end], logo_xs[start:end], c] = \
                        (weights @ sample_vals[:, c]) / w_sum.ravel()
            print(f"  背景: 边界 IDW 插值")
        else:
            non_logo_mask = ~is_logo
            if non_logo_mask.sum() > 0:
                for c in range(3):
                    B_local[is_logo, c] = np.median(mean_I_norm[non_logo_mask, c])
            print(f"  背景: 使用非 logo 区域中位数")

        # 5b. 用局部背景求解 L
        a = alpha.astype(np.float64)
        mask = is_logo & (a > 0.01)
        for c in range(3):
            m = mean_I_norm[:, :, c]
            b_local = B_local[:, :, c]
            L[:, :, c] = np.where(mask,
                                   (m - (1 - a) * b_local) / (a + eps),
                                   m)
            L[:, :, c] = np.clip(L[:, :, c], 0, 1)

        # 6. 完全不透明区保护
        is_solid = (var_I_gray < var_B_est * 0.03) & is_logo
        solid_count = is_solid.sum()
        print(f"  完全不透明区像素: {solid_count}")
        for c in range(3):
            L[:, :, c] = np.where(is_solid, mean_I_norm[:, :, c], L[:, :, c])

    # 7. 边缘抗锯齿平滑
    alpha_smooth = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha = np.where(is_logo, alpha_smooth, 0.0).astype(np.float32)
    alpha = np.clip(alpha, 0, 1)
    L = np.clip(L, 0, 1).astype(np.float32)

    alpha_median = np.median(alpha[is_logo]) if logo_pixel_count > 0 else 0
    low_alpha_ratio = ((alpha > 0.01) & (alpha < 0.5)).sum() / max(logo_pixel_count, 1)
    print(f"\n  === 最终统计 ===")
    print(f"  Alpha 范围 (台标区): [{alpha[is_logo].min():.4f}, {alpha[is_logo].max():.4f}]")
    print(f"  Alpha 中位数 (台标区): {alpha_median:.4f}")
    print(f"  台标区 alpha<0.5 占比: {low_alpha_ratio:.1%}")
    print(f"  台标区 alpha>0.9 占比: {(alpha[is_logo] > 0.9).sum() / max(logo_pixel_count, 1):.1%}")
    print(f"  L 范围 (台标区): [{L[is_logo].min():.4f}, {L[is_logo].max():.4f}]")
    print(f"  L 中位数 (台标区): {np.median(L[is_logo]):.4f}")

    return alpha, L, mean_I_norm.astype(np.float32)


def main():
    print("=== 台标提取 v24 (两级alpha + 阈值0.60 + 比例bbox + 核9×6) ===")
    print(f"流: {STREAM_URL}")
    print(f"帧数: {NUM_FRAMES}, 缩放: {SCALE}")

    # Pass 1: 全帧流式统计，定位 logo
    print("\n[1/3] 全帧流式统计...")
    mean_full, var_full, count1, first_frame = stream_pass(STREAM_URL, NUM_FRAMES, SCALE)
    std_map = np.sqrt(np.clip(var_full, 0, None)).mean(axis=2)
    print(f"  处理了 {count1} 帧")
    print(f"  标准差范围: [{std_map.min():.4f}, {std_map.max():.4f}]")

    # 保存标准差图
    std_vis = np.clip(std_map / (std_map.max() + 1e-8), 0, 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "std_map.png"), (std_vis * 255).astype(np.uint8))

    # 保存第一帧
    cv2.imwrite(os.path.join(OUTPUT_DIR, "first_frame.png"),
                cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # 定位 logo
    bbox_scaled, seg_mask = find_logo_region(std_map)
    x1s, y1s, x2s, y2s = bbox_scaled
    print(f"  Logo 区域 (缩放后): ({x1s},{y1s})-({x2s},{y2s}), 大小={x2s-x1s}x{y2s-y1s}")

    # 还原到原始坐标
    inv_scale = 1.0 / SCALE
    bbox_orig = (int(x1s * inv_scale), int(y1s * inv_scale),
                 int(x2s * inv_scale), int(y2s * inv_scale))
    print(f"  Logo 区域 (原始): ({bbox_orig[0]},{bbox_orig[1]})-({bbox_orig[2]},{bbox_orig[3]})")

    # 画检测框
    vis = first_frame.copy()
    cv2.rectangle(vis, (x1s, y1s), (x2s, y2s), (255, 0, 0), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "detected_region.png"),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "seg_mask.png"), seg_mask * 255)

    del var_full, std_map, seg_mask, std_vis, vis
    gc.collect()

    # Pass 2: ROI 流式统计
    print(f"\n[2/3] ROI 流式统计 (bbox={bbox_scaled})...")
    mean_roi, var_roi, count2, first_roi = stream_pass(STREAM_URL, NUM_FRAMES, SCALE, bbox=bbox_orig)
    print(f"  处理了 {count2} 帧")
    print(f"  ROI 内存占用: {mean_roi.nbytes + var_roi.nbytes + first_roi.nbytes:.0f} bytes")

    # 反解 alpha 和 logo（传入全帧 mean 用于暗度信号计算）
    print("\n[3/3] 反解 alpha 和 logo...")
    alpha, logo_rgb, bg_est = solve_alpha_from_stats(mean_roi, var_roi, mean_full=mean_full)

    # 释放全帧 mean
    del mean_full
    gc.collect()

    h, w = alpha.shape
    print(f"\n  Alpha 范围: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"  非零 alpha 像素 (>0.01): {(alpha > 0.01).sum()}")

    # 放大回原始尺寸
    if SCALE != 1.0:
        orig_h, orig_w = int(alpha.shape[0] / SCALE), int(alpha.shape[1] / SCALE)
        alpha = cv2.resize(alpha, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        logo_rgb = cv2.resize(logo_rgb, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        bg_est = cv2.resize(bg_est, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        h, w = alpha.shape[:2]
        print(f"  放大到原始尺寸: {w}x{h}")

    # 保存结果
    print("\n[保存结果]")
    alpha_uint8 = (alpha * 255).astype(np.uint8)
    alpha_uint16 = (alpha * 65535).astype(np.uint16)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "logo_alpha.png"), alpha_uint16)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "logo_alpha_8bit.png"), alpha_uint8)

    logo_bgr = cv2.cvtColor((logo_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "logo_rgb_8bit.png"), logo_bgr)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = logo_bgr
    rgba[:, :, 3] = alpha_uint8
    cv2.imwrite(os.path.join(OUTPUT_DIR, "logo_rgba.png"), rgba)

    rgba16 = np.zeros((h, w, 4), dtype=np.uint16)
    rgba16[:, :, :3] = (logo_rgb * 65535).astype(np.uint16)
    rgba16[:, :, 3] = alpha_uint16
    cv2.imwrite(os.path.join(OUTPUT_DIR, "logo_rgba_16bit.png"), rgba16)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "bg_estimate.png"),
                cv2.cvtColor((bg_est * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 重合成验证（用第一帧 ROI）
    if first_roi is not None:
        first_roi_f = first_roi.astype(np.float32) / 255.0
        if SCALE != 1.0:
            first_roi_f = cv2.resize(first_roi_f, (w, h), interpolation=cv2.INTER_CUBIC)
        alpha_3c = np.stack([alpha] * 3, axis=2)
        recon = alpha_3c * logo_rgb + (1 - alpha_3c) * bg_est
        diff = np.abs(first_roi_f - recon).mean()
        print(f"  重合成 MAE: {diff:.6f}")

        cmp = np.hstack([first_roi_f, recon, np.clip(np.abs(first_roi_f - recon) * 10, 0, 1)])
        cv2.imwrite(os.path.join(OUTPUT_DIR, "recon_verify.png"),
                    cv2.cvtColor((cmp * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 合成验证：在台标区域上叠加到纯色背景
    print("\n  合成验证...")
    alpha_3c = np.stack([alpha] * 3, axis=2)
    for bg_name, bg_color in [("black", 0), ("white", 255), ("green", (0, 128, 0))]:
        if isinstance(bg_color, int):
            bg = np.full_like(logo_rgb, bg_color, dtype=np.float32)
        else:
            bg = np.full_like(logo_rgb, 0, dtype=np.float32)
            bg[:] = bg_color
        comp = alpha_3c * logo_rgb + (1 - alpha_3c) * bg
        comp_bgr = cv2.cvtColor((np.clip(comp, 0, 1) * 255).astype(np.uint8),
                                cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"composite_{bg_name}.png"), comp_bgr)

    # 信息文件
    with open(os.path.join(OUTPUT_DIR, "logo_info.txt"), "w") as f:
        f.write(f"原视频分辨率: 1920x1080\n")
        f.write(f"Logo 区域 (x1,y1,x2,y2): {bbox_orig[0]},{bbox_orig[1]},{bbox_orig[2]},{bbox_orig[3]}\n")
        f.write(f"Logo 尺寸: {w}x{h}\n")
        f.write(f"提取帧数: {count1}\n")
        f.write(f"缩放比例: {SCALE}\n")
        f.write(f"Alpha 范围: [{alpha.min():.4f}, {alpha.max():.4f}]\n")
        f.write(f"非零 alpha 像素: {(alpha > 0.01).sum()}\n")

    print(f"\n=== 完成 ===")
    print(f"输出在 {OUTPUT_DIR}/:")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fn)
        sz = os.path.getsize(fpath)
        print(f"  {fn:30s} {sz/1024:.1f} KB")


if __name__ == "__main__":
    main()
