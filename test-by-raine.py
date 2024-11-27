import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm, trange
import ffmpeg


def video2tensor(path: str, skip_every: int = 1, merge_every: int = 1):
    cap = cv2.VideoCapture(path)
    frames = []
    target_size = (400, 200)        # width, height

    idx = 0
    while True:
        idx += 1
        ret, frame = cap.read()
        if not ret: break
        if idx % skip_every != 0: continue
        frame = cv2.resize(frame, target_size)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    # merge adjacent frames
    if merge_every > 1:
        merged_frames = []
        for i in range(0, len(frames), merge_every):
            merged_frame = np.mean(frames[i:i+merge_every], axis=0)
            merged_frames.append(merged_frame)
        frames = merged_frames

    tensor = np.array(frames).astype("f4") / 255.0      # [N, H, W, 3], float32, [0, 1]
    return tensor


def blur_batch(images, kernel_size=(5, 5)):
    return np.array([cv2.GaussianBlur(img, kernel_size, 0) for img in images])


def calc_mask(frames: np.ndarray):
    """
    input frames: [N, H, W, 3], float32, [0, 1]
    output masks: [N, H, W], float32, [0, 1]
    """
    diffs: np.ndarray = np.abs(frames - np.roll(frames, shift=-1, axis=0))      # use roll trick
    diffs[-1] = diffs[-2]    # fill the last frame with the second last frame
    masks: np.ndarray = diffs.mean(axis=3)  # [N, H, W]
    near_max = np.quantile(masks, 0.99)
    masks = (masks / near_max).clip(0, 1)   # normalize and clip
    return masks



def visualize_video(frames, masks):
    N, H, W, _ = frames.shape
    for idx in trange(0, N, 1, desc="Generating images"):
        # frame = np.concatenate([frames[idx],masks[idx][:, :, None]], axis=2)
        frame = frames[idx] * masks[idx][:, :, None]
        frame = (frame * 255).astype(np.uint8)
        Image.fromarray(frame).save(f'{output_folder}/{idx:05}.png')


def img_seq_to_video(img_folder, output_path, fps=30):
    (
        ffmpeg
        .input(f'{img_folder}/*.png', pattern_type='glob', framerate=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p', preset='fast', crf=23)
        .overwrite_output()
        .run(quiet=True)
    )


def draw_video_parallel_projection(frames, masks, output_path, dx=1, dy=1):
    N, H, W, _ = frames.shape
    canvas = np.zeros((H + dy * N, W + dx * N, 3), dtype="f4")
    for idx in trange(0, N, 1, desc="Drawing images"):
        frame = frames[idx]
        mask = masks[idx][:, :, None]
        canvas[idx * dy:idx * dy + H, idx * dx:idx * dx + W] = frame * mask + canvas[idx * dy:idx * dy + H, idx * dx:idx * dx + W] * (1 - mask)
    canvas = (canvas * 255).clip(0, 255).astype(np.uint8)

    Image.fromarray(canvas).save(output_path)
    Image.fromarray(canvas).show()


if __name__ == "__main__":
    video_path = "video/yaan.mp4"

    video_name = video_path.split('/')[-1].split('.')[0]
    output_folder = f'output/{video_name}'
    os.makedirs(output_folder, exist_ok=True)

    frames = video2tensor(video_path, skip_every=1, merge_every=10)

    # frames = blur_batch(frames, kernel_size=(3, 3))

    masks = calc_mask(frames)

    threshold = 0.1
    masks[masks < threshold] = 0

    visualize_video(frames, masks)

    # img_seq_to_video(output_folder, f'output/{video_name}.mp4', fps=30)

    draw_video_parallel_projection(frames, masks, f'output/{video_name}_parallel_projection.png', dx=1, dy=1)
