import librosa
import soundfile as sf
import os
import numpy as np
from pathlib import Path

NOISE_LEVEL = 0.02

def split_audio(file_path, output_dir, clip_duration=3, overlap=0.5, sr=16000):
    y, _ = librosa.load(file_path, sr=sr)
    total_samples = len(y)
    clip_len = int(sr * clip_duration)
    step = int(clip_len * (1 - overlap))
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for start in range(0, total_samples - clip_len, step):
        clip = y[start:start + clip_len]
        out_path = os.path.join(output_dir, f"{Path(file_path).stem}_clip_{count}.wav")
        sf.write(out_path, clip, sr)
        count += 1
    print(f"{count}개 클립 생성 완료: {file_path}")

def add_noise(y, level=NOISE_LEVEL):
    return y + level * np.random.randn(len(y))

def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate)

def pitch_shift(y, sr, steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def augment_siren(y, base, output_dir, sr):
    for i in range(100):
        sf.write(f"{output_dir}/{base}_noise_{i}.wav", add_noise(y, level=NOISE_LEVEL), sr)
    sf.write(f"{output_dir}/{base}_slow.wav", time_stretch(y, 0.9), sr)
    sf.write(f"{output_dir}/{base}_fast.wav", time_stretch(y, 1.1), sr)
    sf.write(f"{output_dir}/{base}_pitchup_2.wav", pitch_shift(y, sr, 2), sr)
    sf.write(f"{output_dir}/{base}_pitchdown_2.wav", pitch_shift(y, sr, -2), sr)

def augment_announcement(y, base, output_dir, sr):
    for i in range(50):
        sf.write(f"{output_dir}/{base}_noise_{i}.wav", add_noise(y, level=NOISE_LEVEL), sr)
    sf.write(f"{output_dir}/{base}_slow.wav", time_stretch(y, 0.9), sr)
    sf.write(f"{output_dir}/{base}_fast.wav", time_stretch(y, 1.1), sr)
    sf.write(f"{output_dir}/{base}_pitchup_2.wav", pitch_shift(y, sr, 2), sr)
    sf.write(f"{output_dir}/{base}_pitchdown_2.wav", pitch_shift(y, sr, -2), sr)

def process_class(file_path, class_name, dataset_dir="dataset", sr=16000):
    split_dir = f"{dataset_dir}/{class_name}/splits"
    aug_dir = f"{dataset_dir}/{class_name}/"
    os.makedirs(aug_dir, exist_ok=True)

    # 1. 분할
    split_audio(file_path, split_dir, clip_duration=3, overlap=0.5, sr=sr)

    # 2. 증강
    for f in sorted(os.listdir(split_dir)):
        split_path = os.path.join(split_dir, f)
        y, _ = librosa.load(split_path, sr=sr)
        base = Path(split_path).stem
        sf.write(f"{aug_dir}/{base}_orig.wav", y, sr)

        if class_name == "siren":
            augment_siren(y, base, aug_dir, sr)
        elif class_name == "announcement":
            augment_announcement(y, base, aug_dir, sr)

    # 3. 정리
    for f in os.listdir(split_dir):
        os.remove(os.path.join(split_dir, f))
    os.rmdir(split_dir)
    print(f"✅ 최종 처리 완료: {class_name}")

def process_normal(file_path, class_name, dataset_dir="dataset", sr=16000):
    split_dir = f"{dataset_dir}/{class_name}/splits"
    aug_dir = f"{dataset_dir}/{class_name}/"
    
    # 1. 분할
    split_audio(file_path, split_dir, clip_duration=3, overlap=0.5, sr=sr)

    # 2. 증강
    for f in sorted(os.listdir(split_dir)):
        split_path = os.path.join(split_dir, f)

        y, _ = librosa.load(split_path, sr=sr)
        base = Path(split_path).stem
        os.makedirs(aug_dir, exist_ok=True)
        sf.write(f"{aug_dir}/{base}_orig.wav", y, sr)

    # 3. 분할 파일 제거
    for f in os.listdir(split_dir):
        os.remove(os.path.join(split_dir, f))
    os.rmdir(split_dir)
    print(f"✅ 최종 처리 완료: {class_name}")

if __name__ == '__main__':
    esc50_audio_dir = Path("ESC-50/audio")
    esc50_files = list(esc50_audio_dir.glob("*.wav"))

    # normal 용으로 150개 샘플 랜덤 추출
    selected_normal_files = sorted(np.random.choice(esc50_files, size=400, replace=False))
    for f in selected_normal_files:
        process_normal(str(f), "normal")

    # siren, announcement 클래스 증강 처리
    origin_files = {
        "announcement": "dataset/origin/announcement.wav",
        "siren": ["dataset/origin/siren_1.wav", "dataset/origin/siren_2.wav"]
    }
    for cls, paths in origin_files.items():
        if isinstance(paths, str):
            paths = [paths]
        for p in paths:
            process_class(p, cls)

    # ✅ 증강 파일 개수 출력
    print("\n📊 클래스별 .wav 파일 개수:")
    for cls in ["normal", "announcement", "siren"]:
        cls_dir = Path(f"dataset/{cls}")
        count = len(list(cls_dir.glob("*.wav")))
        print(f"📁 {cls}: {count}개")
