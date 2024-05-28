import os
import soundfile as sf
import librosa
from tqdm import tqdm
from funasr import AutoModel

model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

data_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/noise_fullband"
dest_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/noise_fullband_vad_filtered"
filterout_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/noise_fullband_vad_filterout"
voice_active_thresh_ms = 600


if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
if not os.path.exists(filterout_folder):
    os.makedirs(filterout_folder)

def find_all_wavs(root_dir):
    audio_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

def normalize_filename(filename):
    if ";" in filename:
        filename = filename.replace(";", "_")
    if "&" in filename:
        filename = filename.replace("&", "_")
    if "@" in filename:
        filename = filename.replace("@", "_")
    if "#" in filename:
        filename = filename.replace("#", "_")
    if "$" in filename:
        filename = filename.replace("$", "_")
    if "%" in filename:
        filename = filename.replace("%", "_")
    if "^" in filename:
        filename = filename.replace("^", "_")
    if "(" in filename:
        filename = filename.replace("(", "_")
    if ")" in filename:
        filename = filename.replace(")", "_")
    if "%" in filename:
        filename = filename.replace("%", "_")
    if "+" in filename:
        filename = filename.replace("+", "_")
    if "=" in filename:
        filename = filename.replace("=", "_")
    if '"' in filename:
        filename = filename.replace('"', "_")
    if "'" in filename:
        filename = filename.replace("'", "_")
    if "{" in filename:
        filename = filename.replace("{", "_")
    if "}" in filename:
        filename = filename.replace("}", "_")
    return filename

audio_files = find_all_wavs(data_folder)
# for audio_file in tqdm(audio_files):
for audio_file in tqdm(audio_files):
    try:
        waveform, sample_rate = sf.read(audio_file)
    except:
        print("Error reading file: {}".format(audio_file))
        continue
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=48000)
    segments_result = model.generate(audio_file)
    segments_result = segments_result[0]
    speech_length = 0 # in miliseconds
    if len(segments_result['value']) > 0:
        for segment in segments_result['value']:
            speech_length += segment[1] - segment[0]
    print(f"audio file: {audio_file}, segments: {segments_result['value']}, total length: {speech_length}")
    if speech_length > voice_active_thresh_ms:
        dest_path = audio_file.replace(data_folder, filterout_folder)
        dest_path = normalize_filename(dest_path)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        print("filter out: {}".format(audio_file))
        sf.write(dest_path, waveform, sample_rate)
    else:
        dest_path = audio_file.replace(data_folder, dest_folder)
        dest_path = normalize_filename(dest_path)
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        print("keep: {}".format(audio_file))
        sf.write(dest_path, waveform, sample_rate)
    


