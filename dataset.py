"""
new one
"""
import json
import random
import numpy.random as rnd
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import os
from utils import great_circle_distance, sph2cart, cart2sph, beamformer_max_re, zen_to_ele, azi_to_0_2pi_range, \
    beamformer_max_di
from pathlib import Path


class Dataset(torch.utils.data.Dataset):

    def __init__(self, input_dir, sr=44100, ambiorder=4, angular_window_deg=2.5, ambimode='implicit',
                 dataset='musdb'):

        super().__init__()
        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        self.sr = sr
        self.ambiorder = ambiorder
        self.num_ambi_channels = (ambiorder + 1) ** 2
        self.angular_window = angular_window_deg / 180 * np.pi
        self.ambimode = ambimode
        self.dataset = dataset

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):

        curr_dir = self.dirs[idx]
        # print(curr_dir)

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            metadata = json.load(json_file)

        target_direction, gt_source_angle = get_target_and_gt_direction(metadata, self.angular_window, self.dataset)
        # target_direction read from metadata.json
        # print("t_d: ", target_direction)
        # print("gt: ", gt_source_angle)

        if self.dataset == 'musdb':
            target_source_data, mixed_data = self.get_mixture_and_gt_musdb(
                metadata, curr_dir, target_direction, self.angular_window)
        # get_mixture_and_gt_musdb的返回值为 target_source_data, mixture_waveform
        if self.dataset == 'fuss':
            target_source_data, mixed_data = self.get_mixture_and_gt_fuss(
                metadata, curr_dir, target_direction, self.angular_window)

        # select a subset of ambi channels according to the selected order
        # print("mixed data shpae: ", mixed_data.shape)
        mixed_data = mixed_data[:, 0:self.num_ambi_channels]

        azi_angle_beamformer = azi_to_0_2pi_range(gt_source_angle[0])
        ele_angle_beamformer = zen_to_ele(gt_source_angle[1])
        beamformer_audio = beamformer_max_re(mixed_data, np.array((azi_angle_beamformer, ele_angle_beamformer)))
        # print("max_bf: ", np.max(beamformer_audio))
        # rms = np.sqrt(np.mean(beamformer_audio ** 2))
        # if rms != 0:
        #     beamformer_audio = beamformer_audio * (0.1 / rms)  # desired rms is 0.1


        # print("max_bf_after_norm: ", np.max(beamformer_audio))
        beamformer_audio = torch.tensor(beamformer_audio.T).float()

        # GTs
        target_source_data = np.stack(target_source_data, axis=0)
        target_source_data = np.sum(target_source_data, axis=0)
        target_source_data = torch.unsqueeze(torch.tensor(target_source_data[0, :]).float(), dim=0)

        target_direction[0] = (target_direction[0] + np.pi) / np.pi - 1
        target_direction[1] = 2 * target_direction[1] / np.pi - 1

        conditioning_direction = np.asarray([target_direction[0], target_direction[1]])
        conditioning_direction = torch.tensor(conditioning_direction).float()

        if self.ambimode == 'implicit':
            mixed_data = torch.tensor(mixed_data.T).float()
        elif self.ambimode == 'mixed':
            mixed_data = mixed_data[:, 0:4]  # take the 0 and 1 order
            # print("mixed_data_at_second:", mixed_data.shape)
            mixed_data = torch.tensor(mixed_data.T).float()

        return (mixed_data, target_source_data, conditioning_direction, beamformer_audio)

    def get_mixture_and_gt_musdb(self, metadata, curr_dir, target_direction,
                                 curr_window_size):
        # Iterate over different sources

        target_source_data = []
        for key in ["vocals", "bass", "drums"]:
            gt_audio_files = sorted(list(Path(curr_dir).rglob(key + "_ambi.wav")))
            assert len(gt_audio_files) > 0, "No files found in {}".format(curr_dir)

            source_azi_angle = metadata[key]['panning_angles'][0]
            source_zen_angle = metadata[key]['panning_angles'][1]

            source_azi_angle_beamformer = azi_to_0_2pi_range(source_azi_angle)
            source_ele_angle_beamformer = zen_to_ele(source_zen_angle)

            _, gt_waveform = wavfile.read(gt_audio_files[0])  # (308699, 25)
            # print(gt_waveform)
            gt_waveform = gt_waveform[:, 0:self.num_ambi_channels]

            gt_waveform = beamformer_max_re(gt_waveform,
                                             np.array((source_azi_angle_beamformer, source_ele_angle_beamformer)))

            gt_waveform = gt_waveform.astype(np.float)
            gt_waveform = gt_waveform.T

            # print("lplplp", gt_waveform.shape)

            is_all_zero = np.all((gt_waveform == 0))
            # print("before norm: ", np.max(gt_waveform))
            if not is_all_zero:
                gt_waveform = gt_waveform / (2 ** 15)

            # print(np.max(gt_waveform))
            # print(gt_waveform.shape)

            # gt_waveform = gt_waveform.T.copy()  # shape: 308699
            # gt_waveform = np.expand_dims(gt_waveform, axis=0)  # shape: (1, 308699)

            # Source is inside our target region. Need to save for ground truth.
            if great_circle_distance(source_azi_angle, source_zen_angle, target_direction[0],
                                     target_direction[1]) < curr_window_size:
                target_source_data.append(gt_waveform)
                # [array([[.., .., ..]]), array([[.., .., ..]]), array([[.., .., ..]])]

            # Source is not within our region. Add silence
            else:
                target_source_data.append(np.zeros((gt_waveform.shape[0], gt_waveform.shape[1])))
                # print(len(target_source_data))
        # Load mix
        mix_path = os.path.join(curr_dir, "mix.wav")
        rate, mixture_waveform = wavfile.read(mix_path)
        mixture_waveform = mixture_waveform.astype(np.float)
        mix_is_all_zero = np.all((mixture_waveform[:, 0] == 0))
        if not mix_is_all_zero:
            mixture_waveform = mixture_waveform / (2 ** 15)
            # Normalize mixture
            # mixture_waveform = mixture_waveform / np.amax(np.abs(mixture_waveform[:, 0])) / np.sqrt(
            #     2 * self.ambiorder + 1)
            # print(mixture_waveform)
        # for i in mixture_waveform.T:
        #     print(i)
        # print("attt")
        # print(mixture_waveform[:, 0:4])
        # print("stop point!")
        return target_source_data, mixture_waveform

    def get_mixture_and_gt_fuss(self, metadata, curr_dir, target_direction,
                                curr_window_size):

        target_source_data = []
        for num_source in range(metadata['num_sources']):
            gt_audio_files = sorted(
                list(Path(curr_dir).rglob("source_" + str(num_source) + ".wav")))
            assert len(gt_audio_files) > 0, "No files found in {}".format(
                curr_dir)

            rate, gt_waveform = wavfile.read(gt_audio_files[0])
            gt_waveform = gt_waveform.astype(np.float)
            is_all_zero = np.all((gt_waveform == 0))
            if not is_all_zero:
                rms = np.sqrt(np.mean(gt_waveform ** 2))
                gt_waveform = gt_waveform * (0.1 / rms)  # desired rms is 0.1

            gt_waveform = gt_waveform.T.copy()
            gt_waveform = np.expand_dims(gt_waveform, axis=0)

            source_azi_angle = metadata[str(num_source)]['panning_angles'][0]
            source_zen_angle = metadata[str(num_source)]['panning_angles'][1]

            # Source is inside our target region. Need to save for ground truth
            if great_circle_distance(source_azi_angle, source_zen_angle, target_direction[0],
                                     target_direction[1]) < curr_window_size:
                target_source_data.append(gt_waveform)

            # Source is not within our region. Add silence
            else:
                target_source_data.append(np.zeros((gt_waveform.shape[0], gt_waveform.shape[1])))

        # Load mix
        mix_path = os.path.join(curr_dir, "mix.wav")
        rate, mixture_waveform = wavfile.read(mix_path)
        mixture_waveform = mixture_waveform.astype(np.float)
        mix_is_all_zero = np.all((mixture_waveform[:, 0] == 0))
        if not mix_is_all_zero:
            # Normalize mixture
            mixture_waveform = mixture_waveform / np.amax(np.abs(mixture_waveform[:, 0])) / np.sqrt(
                2 * self.ambiorder + 1)
            # print(mixture_waveform.shape)
            # print(mixture_waveform)

        return target_source_data, mixture_waveform


def get_target_and_gt_direction(metadata, window_size, dataset):
    if dataset == 'musdb':
        # Choose a random source
        random_key = random.choice(["vocals", "bass", "drums"])
        source_azi_angle = metadata[random_key]["panning_angles"][0]  # get azi panning angle
        source_zen_angle = metadata[random_key]["panning_angles"][1]  # get zen panning angle

    if dataset == 'fuss':
        # Choose a random source
        num_sources = metadata['num_sources']
        random_source = random.randint(0, num_sources - 1)
        source_azi_angle = metadata[str(random_source)]["panning_angles"][0]  # get azi panning angle
        source_zen_angle = metadata[str(random_source)]["panning_angles"][1]  # get zen panning angle

    north_pole = np.asarray([0, 0, 1])

    angular_distance = np.abs(rnd.rand() * window_size)

    theta = rnd.rand() * 2 * np.pi
    source_dir = sph2cart(source_azi_angle, source_zen_angle)
    de = np.cross(north_pole, source_dir)
    dn = np.cross(source_dir, de)
    d = dn * np.cos(theta) + de * np.sin(theta)
    b = source_dir * np.cos(angular_distance) + d * np.sin(angular_distance)
    sph = cart2sph(b)

    return [sph[0], sph[1]], [source_azi_angle, source_zen_angle]
