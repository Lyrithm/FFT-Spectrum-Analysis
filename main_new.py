import numpy as np
import librosa
import soundfile as sf
import time
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter  # 确保ImageFilter被导入
from matplotlib import colormaps


def generate_final_spectrogram(
        audio_path,
        output_file='spectrogram_with_axis.png',
        # --- 图像尺寸控制 (水平方向) ---
        output_width_px=12000,
        output_height_px=1600,
        # --- 音频和颜色参数 ---
        time_resolution_ms=5,  # 您现在可以安全地使用高精度
        max_freq_hz=18000,
        dynamic_range_db=65,
        colormap='magma',
        # --- [新增] 安全的后期锐化 ---
        apply_sharpen_filter=True,  # 设为 True 来增强清晰度
        # --- 时间轴外观控制 ---
        axis_height_px=150,
        tick_interval_s=0.5,
        label_rotation_angle=45,
        axis_font_size=32
):
    """
    最终版 (v3.9) - 回归到能正确对齐的v3.6版本，并在此基础上安全地增加后期锐化。

    关键修复:
    1.  [核心修复] 彻底移除了导致数据截断和错位的、错误的降采样逻辑。
    2.  [回归正确] 完全恢复了v3.6版本中稳定、可靠的数据->图像缩放流程。
    3.  [安全锐化] 将锐化作为最后一步，在图像生成后应用，确保不影响坐标对齐。
    """
    print(f"--- 采用最终修复版引擎处理音频: {os.path.basename(audio_path)} ---")
    start_time = time.time()

    # 1. 加载音频和STFT计算 (恢复v3.6的流程)
    try:
        y, sr = sf.read(audio_path, dtype='float32')
        if y.ndim > 1: y = np.mean(y, axis=1)
        duration_s = len(y) / sr
    except Exception as e:
        print(f"文件读取错误: {e}");
        return

    hop_length = int(sr * time_resolution_ms / 1000)
    n_fft = 2048
    S_linear = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S_linear, ref=np.max, amin=1e-5)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freq_mask = freqs <= max_freq_hz
    S_db = S_db[freq_mask, :]

    # 2. [恢复正确逻辑] 使用Pillow对完整的DB数据进行高质量缩放
    temp_img = Image.fromarray(S_db)
    resized_img = temp_img.resize((output_width_px, output_height_px), Image.Resampling.LANCZOS)
    pixel_matrix = np.array(resized_img)

    # 3. 颜色映射
    vmax = pixel_matrix.max()
    vmin = vmax - dynamic_range_db
    normalized_matrix = np.clip((pixel_matrix - vmin) / (vmax - vmin), 0, 1)
    cmap = colormaps.get_cmap(colormap)
    colored_matrix = cmap(normalized_matrix)
    image_array = (colored_matrix[:, :, :3] * 255).astype(np.uint8)

    # 4. 上下翻转并转换为Pillow图像
    image_array_flipped = np.flipud(image_array)
    spectrogram_img = Image.fromarray(image_array_flipped)

    # 5. [新增] 应用安全的后期锐化滤镜
    if apply_sharpen_filter:
        print("正在应用后期锐化滤镜...")
        spectrogram_img = spectrogram_img.filter(ImageFilter.SHARPEN)

    # 6. 创建画布并绘制时间轴 (与v3.6完全相同，保证对齐)
    canvas_width = output_width_px
    canvas_height = output_height_px + axis_height_px
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    canvas.paste(spectrogram_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", size=axis_font_size)
    except IOError:
        font = ImageFont.load_default(size=axis_font_size)

    for t_s in np.arange(0, duration_s, tick_interval_s):
        x_pos = int((t_s / duration_s) * output_width_px)
        draw.line([(x_pos, output_height_px), (x_pos, output_height_px + 20)], fill='black', width=2)
        label_text = f'{int(t_s * 1000 + tick_interval_s * 1000)}'  # 预加刻度值以从0起
        txt_canvas = Image.new('RGBA', (200, 100), (255, 255, 255, 0))
        txt_draw = ImageDraw.Draw(txt_canvas)
        txt_draw.text((0, 0), label_text, font=font, fill='black')
        rotated_txt = txt_canvas.rotate(label_rotation_angle, expand=True, resample=Image.Resampling.BICUBIC)
        canvas.paste(rotated_txt, (x_pos + 5, output_height_px + 25), mask=rotated_txt)
    draw.text((10, canvas_height - 40), "Time (ms)", fill='black', font=font)

    canvas.save(output_file)

    process_time = time.time() - start_time
    print(f"处理完成! 耗时: {process_time:.2f}秒")
    if os.path.exists(output_file):
        output_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"文件大小: {output_size:.2f}MB")


# --- 使用示例 ---
if __name__ == "__main__":
    audio_file = 'audio.flac'
    if not os.path.exists(audio_file):
        print(f"示例音频 '{audio_file}' 不存在，请替换为您自己的文件路径。")
    else:
        generate_final_spectrogram(
            audio_path=audio_file,
            output_file='spectrogram_final_aligned_sharp.png',
            # --- 图像尺寸控制 (水平方向) ---
            output_width_px=48000,
            output_height_px=1080,
            # --- 音频和颜色参数 ---
            time_resolution_ms=1,  # 您现在可以安全地使用高精度
            max_freq_hz=14000,
            dynamic_range_db=65,
            colormap='magma',
            # --- [新增] 安全的后期锐化 ---
            apply_sharpen_filter=True,  # 设为 True 来增强清晰度
            # --- 时间轴外观控制 ---
            axis_height_px=150,
            tick_interval_s=0.5,
            label_rotation_angle=-45,
            axis_font_size=15
        )