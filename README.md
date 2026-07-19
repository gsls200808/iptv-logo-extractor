# IPTV Logo Extractor

从 IPTV 直播流中自动提取带 Alpha 通道的台标 (RGBA PNG)。

支持单流处理和 M3U 播放列表批量处理，已验证 300+ 频道稳定运行。

## 原理

台标叠加在视频画面上遵循物理合成模型：

```
I = alpha * L + (1 - alpha) * B
```

其中 `I` 是观测像素，`L` 是台标颜色，`B` 是背景，`alpha` 是透明度。

利用 Welford 在线算法增量计算多帧的均值和方差，通过方差差异反解 alpha：

```
var_I = (1 - alpha)^2 * var_B
alpha = 1 - std_I / std_B
```

不需要已知背景或台标模板，完全从无监督统计中恢复。

## 文件说明

| 文件 | 说明 |
|------|------|
| `extract_logo_batch.py` | 批量处理脚本，支持 M3U 播放列表 |
| `extract_logo_v3.py` | 单流处理脚本，极低内存版本 |

## 环境依赖

- Python 3.8+
- OpenCV 5.0+（需要 FFmpeg 支持）
- NumPy

```bash
pip install opencv-python numpy
```

## 使用方法

### 批量处理 M3U 播放列表

```bash
python extract_logo_batch.py <M3U_URL或文件路径> [选项]
```

参数说明：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `m3u` | (必填) | M3U 播放列表 URL 或本地路径 |
| `--frames` | 150 | 每个频道采样帧数 |
| `--output` | ./logo_output_batch | 输出目录 |
| `--filter` | 无 | 按频道名过滤（子串匹配） |
| `--limit` | 无 | 最多处理频道数 |
| `--skip-existing` | 关 | 跳过已有输出的频道 |
| `--verbose` | 关 | 显示详细处理日志 |

示例：

```bash
# 处理播放列表，跳过已完成的频道
python extract_logo_batch.py https://example.com/playlist.m3u --skip-existing --verbose

# 只处理包含"CCTV"的频道，最多 10 个
python extract_logo_batch.py playlist.m3u --filter CCTV --limit 10
```

### 单流处理

编辑 `extract_logo_v3.py` 中的 `STREAM_URL` 为目标流地址，然后运行：

```bash
python extract_logo_v3.py
```

## 输出结构

```
logo_output_batch/
  ├── 频道名.png              # 台标 RGBA PNG（透明背景）
  ├── 频道名_2.png            # 同名频道加序号
  ├── 频道名/                 # 调试目录
  │   ├── first_frame.png     # 第一帧截图
  │   ├── detected.png        # 检测框可视化
  │   ├── composite_black.png # 黑底合成验证
  │   └── recon_verify.png    # 重建验证图
  ├── summary.txt             # 处理汇总
  └── progress.log            # 进度日志
```

## 技术细节

### 分辨率自适应

固定 ROI 以 1920x1080 为基准 (0,0)-(460,200)，根据实际视频分辨率等比缩放：

| 分辨率 | ROI 大小 |
|--------|----------|
| 3840x2160 (4K) | (0,0)-(920,400) |
| 1920x1080 (HD) | (0,0)-(460,200) |
| 1280x720 | (0,0)-(307,133) |
| 720x576 (SD) | (0,0)-(172,106) |

### 两遍处理

1. **Pass 1**：全帧 Welford 统计，定位台标区域（低方差连通域）
2. **Pass 2**：ROI 区域精细统计，反解 alpha 并合成台标

### Alpha 策略

- 文字区域：alpha = 1.0（完全不透明）
- 底板区域：alpha 由方差比计算，经幂函数映射平滑
- 边界区域：IDW 插值估计局部背景

## 许可

MIT License
