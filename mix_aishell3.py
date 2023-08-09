"""
------------------------------------------------
Author TungYu Dominick Yeung

For Aishell3 dataset ambisonics generation
------------------------------------------------
"""

import numpy as np
import scipy as sci
import numpy.random as rnd
import scipy.io.wavfile
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
        read_path = os.path.join(root, 'train')
        write_path = os.path.join(base_path, 'train_dir')
    elif subset == 'validate':
        read_path = os.path.join(root, 'validation')
        write_path = os.path.join(base_path, 'validate_dir')
    elif subset == 'test':
        read_path = os.path.join(root, 'eval')
        write_path = os.path.join(base_path, 'test_dir')

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
sampling_rate = 44100
num_samples = length_s * sampling_rate

minimal_angular_dist_rad = float(minimal_angular_dist_deg) / 180 * np.pi
maximal_angular_dist_rad = float(maximal_angular_dist_deg) / 180 * np.pi


print(
    f'Starting dataset generation {dataset}, subset = {subset} \n number of mixes on this node = {num_mixes} '
    f'\n mixes with silent sources = {num_mixes_with_silent_sources} \n sample length = {num_samples} '
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

print("dd")
