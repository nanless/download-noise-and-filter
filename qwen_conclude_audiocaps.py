import os
import pandas as pd
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer

audiocaps_path = "/root/autodl-tmp/codes/ACT/data" # the path to the downloaded audiocaps dataset
wavs_folder = "/root/autodl-tmp/codes/ACT/data/waveforms/train"
csv_file_path = os.path.join(audiocaps_path, "csv_files", "train.csv")
df = pd.read_csv(csv_file_path)

model_name = "Qwen/Qwen2-7B-Instruct"
device = "cuda:0" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    cache_dir="downloaded_models"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="downloaded_models")

for i, row in df.iterrows():
    filename = row["file_name"]
    caption = row["caption"]

    prompt = "The caption of my audio clip is " + caption + ". According to the audio clip caption, is there any human voice in the clip? Answer yes or no only."
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

    print(response)