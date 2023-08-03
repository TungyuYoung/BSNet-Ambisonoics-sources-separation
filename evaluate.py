"""
1. read mix.wav, then beamformer to a single audio
2. beamformer audio -> stft -> beamformer audio fre * mask -> istft -> separated audio
"""
import argparse
import json
import torch
import os
import numpy as np
import scipy.io.wavfile as wavfile
import statistics as stat
from utils import si_sdr, beamformer_max_di, \
    beamformer_max_re, beamformer_max_sdr, zen_to_ele, azi_to_0_2pi_range
from network import TungYu
from pathlib import Path
import seaborn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def predict_audio(model, mixed_data, conditioning_direction, beamformer_audio, args):
    ambi_mix = mixed_data.float().unsqueeze(0)
    ambi_mix = ambi_mix.to(args.device)
    conditioning_direction = conditioning_direction.float().unsqueeze(0)
    conditioning_direction = conditioning_direction.to(args.device)
    beamformer_audio = beamformer_audio.float().unsqueeze(0)
    beamformer_audio = beamformer_audio.to(args.device)

    predict_signal = model(ambi_mix, conditioning_direction, beamformer_audio)
    return predict_signal


def forward_beamformer(bf_type, input_signal, aux):
    if bf_type == 'max_di':
        beamformer = beamformer_max_di
    if bf_type == 'max_re':
        beamformer = beamformer_max_re
    if bf_type == 'max_sdr':
        beamformer = beamformer_max_sdr

    return beamformer(input_signal, aux)


def get_items(dir, ambiorder):
    with open(Path(dir) / 'metadata.json') as json_file:
        metadata = json.load(json_file)

    source_positions = []
    source_audios = []

    for key in sorted(metadata.keys()):
        # get source audio
        gt_audio_files = sorted(list(Path(dir).rglob(key + '.wav')))
        assert (len(gt_audio_files) > 0)
        _, gt_waveform = wavfile.read(gt_audio_files[0])

        gt_waveform = gt_waveform.astype(np.float)
        is_all_zero = np.all((gt_waveform == 0))
        if not is_all_zero:
            rms = np.sqrt(np.mean(gt_waveform ** 2))
            gt_waveform = gt_waveform * (0.1 / rms)

        gt_waveform = gt_waveform.T.copy()
        source_audios.append(gt_waveform)
        source_azi_angle = metadata[key]['panning_angles'][0]
        source_zen_angle = metadata[key]['panning_angles'][1]
        source_positions.append([source_azi_angle, source_zen_angle])

    mix_path = os.path.join(dir, "mix.wav")
    rate, mixture_waveform = wavfile.read(mix_path)
    mixture_waveform = mixture_waveform.astype(np.float)
    mix_is_all_zero = np.all((mixture_waveform[:, 0] == 0))
    if not mix_is_all_zero:
        mixture_waveform = mixture_waveform / np.amax(np.abs(mixture_waveform[:, 0])) / np.sqrt(2 * ambiorder + 1)

    return mixture_waveform, source_positions, source_audios, sorted(metadata.keys())


def evaluate():
    return True


if __name__ == "__main__":
    device = torch.device('cuda')
    model = TungYu(n_audio_channels=5, ambimode="mixed")
    model.load_state_dict(torch.load(
        '/home/tungyu/Project/STFTNet/checkpoints_minidataset/multimic_minidataset_fre_mmse_n_sdr_try_tambi/last.pt'),
        strict=True)
    model.train = False
    model.to(device)
    # target: vocals
    vocals_single_ambi_dir = '/home/tungyu/Project/musdb18/mini_dataset_ambi/test/00000/vocals_ambi.wav'
    mix_25_c = '/home/tungyu/Project/musdb18/mini_dataset_ambi/test/00000/mix.wav'

    _, mix = wavfile.read(mix_25_c)

    print(mix)

    __, vocals_single_ambi = wavfile.read(vocals_single_ambi_dir)

    vocals_position = [1.0036233090036095,
                       1.7836959853851477]

    source_azi_angle = vocals_position[0]
    source_zen_angle = vocals_position[1]

    source_azi_angle_beamformer = azi_to_0_2pi_range(source_azi_angle)
    source_ele_angle_beamformer = zen_to_ele(source_zen_angle)

    beamformer_audio = beamformer_max_re(mix, np.array((source_azi_angle_beamformer, source_ele_angle_beamformer)))

    vocals_single_bf = beamformer_max_re(vocals_single_ambi,
                                        np.array((source_azi_angle_beamformer, source_ele_angle_beamformer)))

    mix = torch.from_numpy(mix.T[0:4])
    print(mix.shape)
    beamformer_audio = torch.from_numpy(beamformer_audio.T)

    mix = mix.to(torch.float32)
    beamformer_audio = beamformer_audio.to(torch.float32)

    mix = mix.to(device)
    beamformer_audio = beamformer_audio.to(device)
    mix = mix.unsqueeze(0)
    beamformer_audio = beamformer_audio.unsqueeze(0)
    output_signal, mask = model(mix, beamformer_audio)

    output_signal = output_signal.cpu().detach().numpy()

    vocals_single_bf = vocals_single_bf
    print(vocals_single_bf)

    print(output_signal)
    print(vocals_single_bf)

    sdr = si_sdr(output_signal[0][0], vocals_single_bf[:, 0], scaling=True, eps=1e-8)
    print(sdr)


    mseloss = F.mse_loss(torch.tensor(output_signal[0][0]), torch.tensor(vocals_single_bf[:, 0]))

    print(mseloss)

    wavfile.write('/home/tungyu/Project/musdb18/mini_dataset_ambi/test/00000/vocals_separated.wav', 44100,
                  output_signal.astype(np.int16))

    wavfile.write('/home/tungyu/Project/musdb18/mini_dataset_ambi/test/00000/vocals_single_bf.wav', 44100,
                  vocals_single_bf.astype(np.int16))

    seaborn.heatmap(mask[0].cpu().detach().numpy())




    plt.show()
    print(output_signal)
