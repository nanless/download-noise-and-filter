import os
import soundfile as sf
import librosa
from tqdm import tqdm

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    model_revision=None,
)

# data_folder = "/data3/zhounan/codes/github_repos/DNS-Challenge/datasets_fullband/noise_fullband/datasets_fullband/noise_fullband"
# dest_folder = "/data3/zhounan/codes/github_repos/DNS-Challenge/datasets_fullband/noise_fullband/datasets_fullband/noise_fullband_modelscope_vad_filtered"
# filterout_folder = "/data3/zhounan/codes/github_repos/DNS-Challenge/datasets_fullband/noise_fullband/datasets_fullband/noise_fullband_modelscope_vad_filterout"
data_folder = "/home/zhounan/data/noise/noise_48k_for_SSI"
dest_folder = "/data2/zhounan/data/noise/noise_48k_for_SSI_filtered"
filterout_folder = "/data2/zhounan/data/noise/noise_48k_for_SSI_filterout"

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
for audio_file in tqdm(audio_files):
    try:
        waveform, sample_rate = sf.read(audio_file)
    except:
        print("Error reading file: {}".format(audio_file))
        continue
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    waveform_16k = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    segments_result = inference_pipeline(audio_in=waveform_16k)
    speech_length = 0
    if len(segments_result) > 0:
        for segment in segments_result['text']:
            speech_length += segment[1] - segment[0]
    if speech_length > 1600:
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
        
    # speech_timestamps = vad_process(audio_file)
    # if len(speech_timestamps) > 0:
    #     print(f"{audio_file} is filtered")
    # else:
    #     os.system(f"cp {audio_file} {dest_folder}")





# def vad_process(audio_file: str):
    
#     pid = multiprocessing.current_process().pid
    
#     with torch.no_grad():
#         wav = read_audio(audio_file, sampling_rate=SAMPLING_RATE)
#         speech_timestamps =  get_speech_timestamps(
#             wav,
#             vad_models[pid],
#             0.46,  # speech prob threshold
#             16000,  # sample rate
#             300,  # min speech duration in ms
#             20,  # max speech duration in seconds
#             600,  # min silence duration
#             512,  # window size
#             200,  # spech pad ms
#         )
#         print(f"{audio_file}: {speech_timestamps}")
#         return speech_timestamps
    

# from concurrent.futures import ProcessPoolExecutor, as_completed

# futures = []
# audio_files = glob.glob(os.path.join(data_folder, '*.wav'))

# with ProcessPoolExecutor(max_workers=NUM_PROCESS, initializer=init_model, initargs=(model,)) as ex:
#     for audio_file in audio_files:
#         futures.append(ex.submit(vad_process, audio_file))

# for finished in as_completed(futures):
#     pprint(finished.result())
# wav = read_audio(wav_path, sampling_rate=SAMPLING_RATE)
# # get speech timestamps from full audio file
# speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)
# print(speech_timestamps)
# # merge all speech chunks to one audio
# save_audio('only_speech.wav',
#            collect_chunks(speech_timestamps, wav), sampling_rate=SAMPLING_RATE)


