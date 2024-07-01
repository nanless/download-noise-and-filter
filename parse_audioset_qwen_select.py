import pandas as pd
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydub import AudioSegment
from tqdm import tqdm
import random
import string

# 配置路径和模型
flac_folder = "/data2/zhounan/data/noise/audioset/data/downloads/audio/bal_train"
model_name = "Qwen/Qwen2-7B-Instruct"
segment_file_path = "/data2/zhounan/data/noise/audioset/data/downloads/balanced_train_segments.csv"
class_pair_file_path = "/data2/zhounan/data/noise/audioset/data/downloads/class_labels_indices.csv"
target_folder = "/data2/zhounan/data/noise/audioset/selected/balanced_train"

def remove_punctuation(input_str):
    return input_str.translate(str.maketrans('', '', string.punctuation))

# 加载类别文件
df = pd.read_csv(class_pair_file_path)

def convert_flac_to_wav(flac_file_path, wav_file_path):
    # 加载FLAC文件
    audio = AudioSegment.from_file(flac_file_path, format="flac")
    # 导出为WAV文件
    audio.export(wav_file_path, format="wav")
    # print(f"Converted {flac_file_path} to {wav_file_path}")

def process_rows(rows, device):
    # 清空当前设备的缓存
    torch.cuda.empty_cache()

    # 加载模型和tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=None,
        trust_remote_code=True,
        cache_dir="downloaded_models"
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="downloaded_models")
    
    for row in tqdm(rows):
        audio_class_dict = {}
        real_classs = []
        classes = row[3:]
        for c in classes:
            temp = c.strip().replace('"', '').replace(" ", "")
            real_class = df[df["mid"] == temp]["display_name"].values[0]
            real_classs.append(real_class.lower())
        audio_class_dict[row[0]] = real_classs
        print(row[0], real_classs)

        prompt = "The classes of one audio clip assigned by human annotators is " + ",".join(real_classs) + ". According to the audio clip classes, is there any human voice in the clip? Answer yes or no only."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        ###去掉标点
        response = response.lower().replace(".", "").replace(",", "").replace("?", "").replace("!", "")

        # print(response)

        if response == "no":
            selected_class = random.choice(real_classs)
            print("Randomly selected class:", selected_class)

            ##去掉任何标点

            response = remove_punctuation(selected_class.lower())

            response = "_".join(response.split())

            print("Target folder:", os.path.join(target_folder, response))
            
            if not os.path.exists(os.path.join(target_folder, response)):
                os.makedirs(os.path.join(target_folder, response))
            
            if os.path.exists(os.path.join(flac_folder, row[0] + ".flac")):
                convert_flac_to_wav(os.path.join(flac_folder, row[0] + ".flac"), os.path.join(target_folder, response, row[0] + ".wav"))

        # 清空缓存以防止显存溢出
        torch.cuda.empty_cache()

# 读取数据
with open(segment_file_path, "r") as f:
    reader = list(csv.reader(f))[3:]  # 跳过前3行

# 处理数据
if __name__ == "__main__":
    device = "cuda:0"
    process_rows(reader, device)
