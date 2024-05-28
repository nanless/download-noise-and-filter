import os
import random
import shutil

dest_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/noise_fullband_vad_filtered"
filterout_folder = "/root/autodl-tmp/DNS_challenge5_data/datasets_fullband/noise_fullband_vad_filterout"

select_number = 100
select_dest_folder = f"{dest_folder}_selected"
select_filterout_folder = f"{filterout_folder}_selected"

os.makedirs(select_dest_folder, exist_ok=True)
os.makedirs(select_filterout_folder, exist_ok=True)

def walk_all_wavs(folder):
    filepaths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                filepaths.append(os.path.join(root, file))
    return filepaths

def copy_selected_files(source_folder, destination_folder, number_of_files):
    all_wavs = walk_all_wavs(source_folder)
    random.shuffle(all_wavs)
    for wavfile in all_wavs[:number_of_files]:
        destination_file = os.path.join(destination_folder, os.path.basename(wavfile))
        shutil.copyfile(wavfile, destination_file)

copy_selected_files(dest_folder, select_dest_folder, select_number)
copy_selected_files(filterout_folder, select_filterout_folder, select_number)
