import os
import soundfile as sf
import librosa
from tqdm import tqdm

data_folder = "output/download_noises"
dest_folder = "output/download_noises_cutto20s"
segment_seconds = 20
target_sample_rate = 48000
target_sample_length = int(segment_seconds * target_sample_rate)

def find_all_wavs(root_dir):
    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append((os.path.join(root, file), root))
    return audio_files

def normalize_filename(filename):
    for char in [';', '&', '@', '#', '$', '%', '^', '(', ')', '+', '=', '"', "'", "{", "}"]:
        filename = filename.replace(char, "_")
    return filename

audio_files = find_all_wavs(data_folder)

for audio_file, original_dir in tqdm(audio_files):
    try:
        waveform, sample_rate = sf.read(audio_file)
        if waveform.ndim > 1:
            waveform = waveform[:, 0]
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate, res_type="fft")
    except Exception as e:
        print("Error reading file: {}".format(audio_file), e)
        continue

    relative_path = os.path.relpath(original_dir, data_folder)
    save_dir = os.path.join(dest_folder, relative_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_samples = len(waveform)
    num_segments = num_samples // target_sample_length

    base_filename = os.path.basename(audio_file)
    base_filename = normalize_filename(base_filename)

    for segment in range(num_segments):
        start = segment * target_sample_length
        end = start + target_sample_length
        segment_waveform = waveform[start:end]
        segment_filename = f"{base_filename}_segment{segment}.wav"
        segment_path = os.path.join(save_dir, segment_filename)
        sf.write(segment_path, segment_waveform, target_sample_rate)

    # 如果最后一个片段长度不满20秒但大于10秒仍需要保存
    if num_samples % target_sample_length > target_sample_length / 2:
        start = num_segments * target_sample_length
        segment_waveform = waveform[start:]
        segment_filename = f"{base_filename}_segment{num_segments}.wav"
        segment_path = os.path.join(save_dir, segment_filename)
        sf.write(segment_path, segment_waveform, target_sample_rate)
