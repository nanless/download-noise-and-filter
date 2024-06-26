from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
from tqdm import tqdm
import shutil

def walk_wav_files(folder):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.wav'):
                yield os.path.join(root, file)


original_folder = "/root/autodl-tmp/codes/ACT/data/waveforms"
target_folder = "/root/autodl-tmp/data/noise/audiocaps_filtered"

os.makedirs(target_folder, exist_ok=True)
torch.manual_seed(1234)

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True, cache_dir="downloaded_models")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True, cache_dir="downloaded_models").eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)


for wav_file in tqdm(list(walk_wav_files(original_folder))):

    # 1st dialogue turn
    query = tokenizer.from_list_format([
        {'audio': wav_file}, # Either a local path or an url
        {'text': 'describe the aoustic environment and acoustic events in the audio in detail'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    # print(response)
    response, history = model.chat(tokenizer, 'is there any people talking or any human sounds in the audio? just answer yes or no', history=history)
    print(response)
    if "no" in response.lower():
        
        response, history = model.chat(tokenizer, "describe the aoustic environment and acoustic events in the audio with only one English word, don't use any abbreviations or acronyms or any punctuation", history=history)
        response, history = model.chat(tokenizer, "conclude with only one English word, don't use any punctuation", history=history)
        # print(response)
        ###去掉标点符号，用空格分隔
        response = response.replace(".", "").replace(",", "").replace("!", "").replace("?", "").replace(":", "").replace(";", "").replace("-", " ")
        response = response.strip().lower()
        ###把几个词用_连接起来，比如：quiet_room_with_people_talking
        noise_category = "_".join(response.split())
        # print(noise_category)
        os.makedirs(os.path.join(target_folder, noise_category), exist_ok=True)
        shutil.copy(wav_file, os.path.join(target_folder, noise_category, os.path.basename(wav_file)))

        