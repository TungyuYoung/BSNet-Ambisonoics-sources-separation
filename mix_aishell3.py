"""
------------------------------------------------
Author TungYu Dominick Yeung

For Aishell3 dataset ambisonics generation
------------------------------------------------

python mix_aishell3.py train 15000 0 5 /home/tungyu/Project/datasets/aishell3/BS_dataset --dataset aishell3 --render_room --room_size_range 2 2 1 --rt_range 0.2
"""

import numpy as np
import scipy as sci
import numpy.random as rnd
import scipy.io.wavfile as wavfile
import musdb
import argparse
import copy
from pathlib import Path
import os
import json
import librosa
from utils import eval_sh
from pyroom.pyroom import *
from pyroom.pyroom import Roomsimulator
from utility.Coordinates import *
from utility.Coordinates import Coordinates


def generateOneRandomSourcePosition(roomSize):
    position = roomSize.cart * (np.random.rand(3) * 2 - 1)
    position[2] = roomSize.cart[2] * (np.random.rand(1) - 0.5)
    c = Coordinates(position)

    return c


def prepareAISHELL3():
    root = '../datasets/aishell3/'  # path to FUSS ssdata

    if subset == 'train':
        read_path = os.path.join(root, 'train/wav')
        write_path = os.path.join(base_path, 'train')
    elif subset == 'validate':
        read_path = os.path.join(root, 'validation/wav')
        write_path = os.path.join(base_path, 'validate')
    elif subset == 'test':
        read_path = os.path.join(root, 'eval/wav')
        write_path = os.path.join(base_path, 'test')

    return read_path, write_path


#  Parsers define
parser = argparse.ArgumentParser()
parser.add_argument("subset", help="subset is train, validate or test")
parser.add_argument("num_mixes", help="number of mixes created on one instance", type=int)  # 10000
parser.add_argument("num_mixes_with_silent_sources",
                    help="number of mixes with silent sources created on that instance", type=int)  # 3000
parser.add_argument("minimal_angular_dist", help="minimum angular distance between sources in degree", type=float)  # 5°
parser.add_argument("base_path", help="path for the resulting dataset")

parser.add_argument("--maximal_angular_dist",
                    help="maximal angular distance between sources in degree (for generating closed sources dataset)",
                    type=float, default=180.0)
parser.add_argument("--batch_index", help="when running on multiple instances, this is the index of the instance",
                    type=int, default=0)
parser.add_argument('--render_room', dest='render_room', action='store_true', default=False)
parser.add_argument("--dataset", help="for now, musdb or fuss", type=str, default='musdb')
parser.add_argument("--level_threshold_db", help="level threshold db for a mix not to count as silent", type=float,
                    default='-60.0')

parser.add_argument("--room_size_range", help="range of variation from the default room size in m", type=list,
                    default=[0, 0, 0], nargs='+')
parser.add_argument("--rt_range", help="range of variation from the default reverberation time", type=float,
                    default=0)

args = parser.parse_args()

subset = args.subset
num_mixes = args.num_mixes
num_mixes_with_silent_sources = args.num_mixes_with_silent_sources
minimal_angular_dist_deg = args.minimal_angular_dist
maximal_angular_dist_deg = args.maximal_angular_dist
base_path = args.base_path
batch_index = args.batch_index

render_room = args.render_room
room_size_range = np.array(args.room_size_range[0]).astype(np.float64)
rt_range = args.rt_range

level_threshold_db = args.level_threshold_db

dataset = args.dataset

max_order = 4
num_sh_channels = (max_order + 1) ** 2
length_s = 6
ir_length_s = 1

seed_value = 13
rnd.seed(seed_value)

if dataset == 'aishell3':
    sampling_rate = 44100
    num_samples = length_s * sampling_rate

minimal_angular_dist_rad = float(minimal_angular_dist_deg) / 180 * np.pi
maximal_angular_dist_rad = float(maximal_angular_dist_deg) / 180 * np.pi

print(
    f'Starting dataset generation {dataset}, subset = {subset} \n number of mixes on this node = {num_mixes} '
    f'\n mixes with silent sources = {num_mixes_with_silent_sources} \n sample length = {6 * 44100} '
    f'\n result path = {base_path} \n room rendering {render_room}')

if render_room:
    roomSim = Roomsimulator()

    # Default Room Size
    default_room_size = Coordinates([3, 4, 3])

    # Default Reverberation time for [  125.   250.   500.  1000.  2000.  4000.  8000. 16000.] Hz
    default_rt = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

    roomSim.fs = sampling_rate
    roomSim.maxShOrder = max_order
    # set some parameters
    roomSim.maxIsOrder = 6

    # prepare general room simulation
    roomSim.prepareImageSource()
    roomSim.prepareWallFilter()
    roomSim.plotWallFilters()

    roomSim.irLength_s = ir_length_s

    roomSim.alignDirectSoundToStart = True

iMix = 0
iMixWithSilentSources = 0

read_path, write_path = prepareAISHELL3()
# print(read_path, write_path)

# prepare file_address_list, and then select 3 audios randomly from it to generate dataset
file_address_list = []
for root, subdirectories, files in os.walk(read_path):
    aux_dir = {}
    for subdirectory in subdirectories:
        # print(os.path.join(root, subdirectory))
        curr_path = os.path.join(root, subdirectory)
        # print(curr_path)
        for filename in os.listdir(curr_path):
            file_address = os.path.join(curr_path, filename)
            file_address_list.append(file_address)

rnd.shuffle(file_address_list)  # shuffle the list
num_group = int(len(file_address_list) / 3)

for i in range(num_group):
    speaker_1 = file_address_list[i * 3]
    speaker_2 = file_address_list[i * 3 + 1]
    speaker_3 = file_address_list[i * 3 + 2]

    _, speaker_1_data = wavfile.read(speaker_1)
    _, speaker_2_data = wavfile.read(speaker_2)
    _, speaker_3_data = wavfile.read(speaker_3)

    hope_data_length = length_s * sampling_rate

    speaker_1_length = len(speaker_1_data)
    speaker_2_length = len(speaker_2_data)
    speaker_3_length = len(speaker_3_data)

    speaker_1_hope = np.zeros(hope_data_length)
    speaker_2_hope = np.zeros(hope_data_length)
    speaker_3_hope = np.zeros(hope_data_length)

    longest_length = max(speaker_1_length, speaker_2_length, speaker_3_length)

    if len(speaker_1_data) < hope_data_length:
        sp1_start = rnd.randint(0, hope_data_length - len(speaker_1_data))
        speaker_1_hope[sp1_start:sp1_start + len(speaker_1_data)] += speaker_1_data
    else:
        speaker_1_data = speaker_1_data[0:hope_data_length]
        speaker_1_hope = speaker_1_hope + speaker_1_data

