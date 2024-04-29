# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

# This is bash script actually run the downloading and resampling script.
# See instructions in freesound_download.py

# Change these arguments if you want
page_size=100   # Number of sounds per page
max_samples=300 # Maximum number of sound samples
min_filesize=0  # Minimum filesize allowed (in MB)
max_filesize=100 # Maximum filesize allowed (in MB)

if [[ $# -ne 3 ]]; then
  echo "Require number of all files | data directory | resample data directory as arguments to the script"
  exit 2
fi

NUM_ALL_FILES=$1
DATADIR=$2
RESAMPLE_DATADIR=$3

if [ ! -d "$DATADIR" ]; then
    echo "Creating dir $DATADIR"
    mkdir -p "$DATADIR"
fi

if [ ! -d "$RESAMPLE_DATADIR" ]; then
    echo "Creating dir $RESAMPLE_DATADIR"
    mkdir -p "$RESAMPLE_DATADIR"
fi

# Merging and deduplicating categories while converting to lowercase
declare -a categories=(
  "TV background noise"
  "babble"
  "race"
  "mouse"
  "chair"
  "renovation"
  "keyboard"
  "street"
  "office"
  "traffic"
  "wind"
  "engine"
  "restaurant"
  "cafe"
  "brake"
  "domestic"
  "city"
  "microwaveoven"
  "door"
  "distortion"
  "tape hiss"
  "hubbub"
  "vibration"
  "bell"
  "cellphone"
  "ring"
  "alarm"
  "cacophony"
  "hit"
  "scratch"
  "snap"
  "clap"
  "tennis"
  "basketball"
  "ping-pong"
  "guitar"
  "drum"
  "bass"
  "violin"
  "piano"
  "click"
  "knock"
  "echo"
  "rural"
  "nature"
  "urban"
  "manmade"
  "electronic"
  "car"
  "bus"
  "truck"
  "ambulance"
  "motor"
  "aircraft"
  "helicopter"
  "bicycle"
  "skateboard"
  "subway"
  "metro"
  "rail"
  "train"
  "boat"
  "ship"
  "bird"
  "dog"
  "cat"
  "horse"
  "animal"
  "park"
  "forest"
  "mountain"
  "drone"
  "machine"
  "horn"
  "music"
  "rain"
  "thunder"
  "factory"
  "home"
  "school"
  "church"
  "hospital"
  "bar"
  "nightclub"
  "noise"
  "aircondition"
  "airplane"
  "airport"
  "fan"
  "fire"
  "fireworks"
  "water"
  "road"
  "station"
  "toilet"
  "river"
  "walk"
  "pink"
  "white"
  "bounce"
  "rattle"
  "insect"
  "market"
  "rub"
  "blow"
  "crowd"
  "club"
  "pat"
  "footstep"
  "call"
  "cheer"
  "storm"
  "applause"
  "kitchen"
  "clock"
  "frying"
  "drop"
  "blow mic"
  "whistle"
  "motorcycle"
  "racecar"
  "sink"
  "siren"
  "chatter"
  "squeak"
  "rustle"
  "hum"
  "whir"
  "buzz"
  "beep"
  "screech"
  "howl"
  "rumble"
  "crackle"
  "splash"
  "bang"
  "clatter"
  "shatter"
  "buzzer"
  "whine"
  "drip"
  "fizz"
  "coffee maker"
  "vacuum cleaner"
  "garbage disposal"
  "printer"
  "elevator ding"
  "escalator"
  "air conditioner"
  "washing machine"
  "dishwasher"
  "electric toothbrush"
  "lawnmower"
  "garage door"
  "leaf blower"
  "shopping cart"
  "cash register"
  "children playing"
  "basketball bounce"
  "dogs barking"
  "fire alarm"
  "ice machine"
  "kettle boiling"
  "fridge humming"
  "microwave beeping"
  "stove clicking"
  "blender whirring"
  "doorbell ringing"
  "phone ringing"
  "tablet tapping"
  "computer keyboard typing"
  "mouse clicking"
  "printer scanning"
  "fax machine transmitting"
  "photocopier operating"
  "radio broadcasting"
  "game console buzzing"
  "digital assistant speaking"
  "home security alarm"
  "window knocking"
  "floor creaking"
)

# categories=($(printf "%s\n" "${categories[@]}" | awk '{print tolower($0)}' | sort -u))

# categories=($(printf "%s\n" "${categories[@]}" | awk '{print tolower($0)}'))

WAV_FILECOUNT="$(find $DATADIR  -name '*.wav' -type f | wc -l)"
FLAC_FILECOUNT="$(find $DATADIR  -name '*.flac' -type f | wc -l)"
FILECOUNT="$((WAV_FILECOUNT + FLAC_FILECOUNT))"
echo "File count: " $FILECOUNT

while((FILECOUNT <= NUM_ALL_FILES))
do
  for category in "${categories[@]}"
  do
    
    python freesound_download.py --data_dir "${DATADIR}" --category "${category}" --page_size "${page_size}" --max_samples "${max_samples}" --min_filesize "${min_filesize}" --max_filesize "${max_filesize}"
    ret=$?
    if [ $ret -ne 0 ]; then
        exit 1
    fi
  done

  WAV_FILECOUNT="$(find $DATADIR  -name '*.wav' -type f | wc -l)"
  FLAC_FILECOUNT="$(find $DATADIR  -name '*.flac' -type f | wc -l)"
  FILECOUNT="$((WAV_FILECOUNT + FLAC_FILECOUNT))"
  echo "Current file count is: " $FILECOUNT
done

# RESAMPLE
echo "Got enough data. Start resample!"
python freesound_resample.py --data_dir="${DATADIR}" --resampled_dir="${RESAMPLE_DATADIR}" --sample_rate=48000

echo "Done resample data!"
