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

    if subset == 'train':  # 19831
        read_path = os.path.join(root, 'train/wav')
        write_path = os.path.join(base_path, 'train')
    elif subset == 'validate':  # 1257
        read_path = os.path.join(root, 'validation/wav')
        write_path = os.path.join(base_path, 'validate')
    elif subset == 'test':  # 8285
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

    speaker_1_track = speaker_1.split("/")[-1].split(".")[0]
    speaker_2_track = speaker_2.split("/")[-1].split(".")[0]
    speaker_3_track = speaker_3.split("/")[-1].split(".")[0]

    speaker_tracks = [speaker_1_track, speaker_2_track, speaker_3_track]

    speaker_data = {
        speaker_1_track: [],
        speaker_2_track: [],
        speaker_3_track: []
    }

    _, speaker_1_data = wavfile.read(speaker_1)
    speaker_data[speaker_1_track] = speaker_1_data
    _, speaker_2_data = wavfile.read(speaker_2)
    speaker_data[speaker_2_track] = speaker_2_data
    _, speaker_3_data = wavfile.read(speaker_3)
    speaker_data[speaker_3_track] = speaker_3_data

    combined = [(track, speaker_data[track]) for track in speaker_tracks]

    sorted_combined = sorted(combined, key=lambda x: len(x[1]))

    speaker_tracks = [item[0] for item in sorted_combined]

    speaker_data_group = []

    for track in speaker_tracks:
        ddd = speaker_data[track]
        speaker_data_group.append(ddd)

    hope_data_length = length_s * sampling_rate

    target_1_hope = np.zeros(hope_data_length)
    target_2_hope = np.zeros(hope_data_length)
    target_3_hope = np.zeros(hope_data_length)

    target_1_data = None
    target_2_data = None
    target_3_data = None

    if len(speaker_data_group[2]) < hope_data_length:  # 1 is the longest
        sp1_start = rnd.randint(0, hope_data_length - len(speaker_data_group[2]))
        target_1_hope[sp1_start:sp1_start + len(speaker_data_group[2])] += speaker_data_group[2]
        target_2_hope[sp1_start:sp1_start + len(speaker_data_group[1])] += speaker_data_group[1]
        target_3_hope[sp1_start:sp1_start + len(speaker_data_group[0])] += speaker_data_group[0]
    elif len(speaker_data_group[1]) < hope_data_length:
        target_1_data = speaker_data_group[2][0:hope_data_length]
        target_1_hope = target_1_hope + target_1_data
        sp2_start = rnd.randint(0, hope_data_length - len(speaker_data_group[1]))
        target_2_hope[sp2_start:sp2_start + len(speaker_data_group[1])] += speaker_data_group[1]
        target_3_hope[sp2_start:sp2_start + len(speaker_data_group[0])] += speaker_data_group[0]
    elif len(speaker_data_group[0]) < hope_data_length:
        target_1_data = speaker_data_group[2][0:hope_data_length]
        target_1_hope = target_1_hope + target_1_data
        target_2_data = speaker_data_group[1][0:hope_data_length]
        target_2_hope = target_2_hope + target_2_data
        sp3_start = rnd.randint(0, hope_data_length - len(speaker_data_group[0]))
        target_3_hope[sp3_start:sp3_start + len(speaker_data_group[0])] += speaker_data_group[0]
    else:
        target_1_data = speaker_data_group[2][0:hope_data_length]
        target_1_hope = target_1_hope + target_1_data
        target_2_data = speaker_data_group[1][0:hope_data_length]
        target_2_hope = target_2_hope + target_2_data
        target_3_data = speaker_data_group[0][0:hope_data_length]
        target_3_hope = target_3_hope + target_3_data

    # Now we have 3 target_hope list to emerge to become a mixture wav

    # rendering room
    if render_room:
        # modify room size and reverberation time on each iteration
        room_size = Coordinates(default_room_size.cart + room_size_range * (np.random.rand(3) - 0.5) * 2)
        roomSim.roomSize = room_size
        # get the maximal possible source position range
        distance_from_walls_m = 0.2
        source_position_range = Coordinates(room_size.cart / 2.0 - distance_from_walls_m)

        roomSim.rt = default_rt + rt_range * (np.random.rand(8) - 0.5) * 2

    else:
        source_position_range = Coordinates(
            [1, 1, 1])  # in case there is no room simulations, generate points in a cube

    p1 = generateOneRandomSourcePosition(source_position_range)
    p2 = copy.deepcopy(p1)
    p3 = copy.deepcopy(p2)

    while (p1.greatCircleDistanceTo(p2) < minimal_angular_dist_rad or p1.greatCircleDistanceTo(p2)
           > maximal_angular_dist_rad):
        p2 = generateOneRandomSourcePosition(source_position_range)

    # Try placing another source, at least minimal_angular_dist_rad away from the first two
    while (p1.greatCircleDistanceTo(p3) < minimal_angular_dist_rad) or p1.greatCircleDistanceTo(
            p3) > maximal_angular_dist_rad or \
            (p2.greatCircleDistanceTo(p3) < minimal_angular_dist_rad or p2.greatCircleDistanceTo(
                p3) > maximal_angular_dist_rad):
        p3 = generateOneRandomSourcePosition(source_position_range)

    if render_room:
        ir_length_samp = ir_length_s * sampling_rate
        # first source
        roomSim.sourcePosition = p1
        srir1 = roomSim.simulate()
        x_source_1_ambi = np.zeros((hope_data_length + srir1.shape[0] - 1, num_sh_channels))
        for iShChannel in range(num_sh_channels):
            x_source_1_ambi[:, iShChannel] = sci.signal.convolve(target_1_hope, srir1[:, iShChannel])

        # second source
        roomSim.sourcePosition = p2
        srir2 = roomSim.simulate()
        x_source_2_ambi = np.zeros((hope_data_length + srir2.shape[0] - 1, num_sh_channels))
        for iShChannel in range(num_sh_channels):
            x_source_2_ambi[:, iShChannel] = sci.signal.convolve(target_2_hope, srir2[:, iShChannel])

        # third source
        roomSim.sourcePosition = p3
        srir3 = roomSim.simulate()
        x_source_3_ambi = np.zeros((hope_data_length + srir3.shape[0] - 1, num_sh_channels))
        for iShChannel in range(num_sh_channels):
            x_source_3_ambi[:, iShChannel] = sci.signal.convolve(target_3_hope, srir3[:, iShChannel])

        x_source_1_mono = np.hstack((target_1_hope, np.zeros(ir_length_samp - 1)))
        x_source_2_mono = np.hstack((target_2_hope, np.zeros(ir_length_samp - 1)))
        x_source_3_mono = np.hstack((target_3_hope, np.zeros(ir_length_samp - 1)))

        x_mix = (x_source_1_ambi + x_source_2_ambi + x_source_3_ambi) / 3
        x_mix_max_jk = np.max(np.abs(x_mix))
        x_mix = x_mix / x_mix_max_jk

        x_source_1_mono = x_source_1_mono / np.max(np.abs(x_source_1_mono))
        x_source_2_mono = x_source_2_mono / np.max(np.abs(x_source_2_mono))
        x_source_3_mono = x_source_3_mono / np.max(np.abs(x_source_3_mono))

        x_source_1_ambi = x_source_1_ambi / np.max(np.abs(x_source_1_ambi))
        x_source_2_ambi = x_source_2_ambi / np.max(np.abs(x_source_2_ambi))
        x_source_3_ambi = x_source_3_ambi / np.max(np.abs(x_source_3_ambi))

        output_prefix_dir = os.path.join(write_path,
                                         str(iMix))  # write path: /home/tungyu/Project/datasets/BS_dataset/train
        Path(output_prefix_dir).mkdir(parents=True, exist_ok=True)

        # write the output wav
        output_mix_dir = os.path.join(output_prefix_dir, 'mix.wav')
        wavfile.write(output_mix_dir, sampling_rate, x_mix)

        output_source_1_ambi = os.path.join(output_prefix_dir, 'speaker_3_ambi.wav')
        wavfile.write(output_source_1_ambi, sampling_rate, x_source_1_ambi)

        output_source_2_ambi = os.path.join(output_prefix_dir, 'speaker_2_ambi.wav')
        wavfile.write(output_source_2_ambi, sampling_rate, x_source_2_ambi)

        output_source_3_ambi = os.path.join(output_prefix_dir, 'speaker_1_ambi.wav')
        wavfile.write(output_source_3_ambi, sampling_rate, x_source_3_ambi)

        output_source_1_mono = os.path.join(output_prefix_dir, 'speaker_3_mono.wav')
        wavfile.write(output_source_1_mono, sampling_rate, x_source_1_mono)

        output_source_2_mono = os.path.join(output_prefix_dir, 'speaker_2_mono.wav')
        wavfile.write(output_source_2_mono, sampling_rate, x_source_2_mono)

        output_source_3_mono = os.path.join(output_prefix_dir, 'speaker_1_mono.wav')
        wavfile.write(output_source_3_mono, sampling_rate, x_source_3_mono)

        output_source_srir_1 = os.path.join(output_prefix_dir, 'srir3.wav')
        srir_int = srir1
        wavfile.write(output_source_srir_1, sampling_rate, srir_int)

        output_source_srir_2 = os.path.join(output_prefix_dir, 'srir2.wav')
        srir_int = srir2
        wavfile.write(output_source_srir_2, sampling_rate, srir_int)

        output_source_srir_3 = os.path.join(output_prefix_dir, 'srir1.wav')
        srir_int = srir3
        wavfile.write(output_source_srir_3, sampling_rate, srir_int)

        azi = np.array([p1.azi, p2.azi, p3.azi])
        zen = np.array([p1.zen, p2.zen, p3.zen])

        azi_normalized = (azi + np.pi) % (2 * np.pi) - np.pi
        dir_sph = np.vstack((azi_normalized, zen))

        metadata = {}

        metadata[speaker_tracks[0]] = {
            'panning_angles': dir_sph[:, 2].tolist(),
            'position_cartesian': p3.cart.tolist(),
        }

        metadata[speaker_tracks[1]] = {
            'panning_angles': dir_sph[:, 1].tolist(),
            'position_cartesian': p2.cart.tolist(),
        }

        metadata[speaker_tracks[2]] = {
            'panning_angles': dir_sph[:, 0].tolist(),
            'position_cartesian': p1.cart.tolist(),
        }

        metadata_file = str(Path(output_prefix_dir) / "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

        room_metadata = {}
        room_metadata['room'] = {
            'room_size': roomSim.roomSize.cart.tolist(),
            'rt': roomSim.rt.tolist(),
        }
        room_metadata_file = str(Path(output_prefix_dir) / "room_metadata.json")
        with open(room_metadata_file, "w") as f:
            json.dump(room_metadata, f, indent=4)

    iMix = iMix + 1
    print('iMix: ' + str(iMix))
    print('\n')
