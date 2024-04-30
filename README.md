# download-noise-and-filter
Download FreeSound noise and filter based on VAD model

1. We will need some requirements including freesound, requests, requests_oauthlib, joblib, librosa and sox. If they are not installed, please run `pip install -r requirements.txt`
2. Create an API key for freesound.org at  https://freesound.org/help/developers/
3. Create a python file called `freesound_private_apikey.py` and add lined `api_key = <your Freesound api key>` and `client_id = <your Freesound client id>`
4. Authorize by run `python freesound_download.py --authorize` and visit website, and paste response code
5. Feel free to change any arguments in download_resample_freesound.sh such as max_samples and max_filesize
6. Run `bash download_resample_freesound.sh <numbers of files you want> <download data directory> <resampled data directory>`
7. change parameter settings in `cut_audio.py`, then `python cut_audio.py` to cut the downloaded noise files to 20-second long noise files.
8. change paremeter settings in `funasr_vad_detect.py`, then `python funasr_vad_detect.py` to filter out noise files based on voice activity detection.
