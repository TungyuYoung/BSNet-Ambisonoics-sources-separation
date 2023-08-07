import argparse
import json
import torch
import os
import numpy as np
import scipy.io.wavfile as wavfile
import statistics as stat
from utils import si_sdr, beamformer_max_di, \
    beamformer_max_re, beamformer_max_sdr, zen_to_ele, azi_to_0_2pi_range, si_sdr_torch_edition
from network import TungYu
from pathlib import Path


def flatten(t):
    return [item for sublist in t for item in sublist]


def forward_pass(model, mixed_data, beamformed_data, device):
    ambi_mixes = mixed_data.float().unsqueeze(0)
    ambi_mixes = ambi_mixes.to(device)
    beamformed_data = beamformed_data.T.float().unsqueeze(0)
    beamformed_data = beamformed_data.to(device)
    output_signal, _ = model(ambi_mixes, beamformed_data)
    return output_signal


def get_waveform_n_angle(curr_dir):
    ambiorder = 4
    with open(Path(curr_dir) / 'metadata.json') as json_file:
        metadata = json.load(json_file)
    # print(metadata)
    source_positions = []
    source_audios = []
    for key in sorted(metadata.keys()):
        # print(key)
        source_audio_files = sorted(list(Path(curr_dir).rglob(key + '.wav')))
        # print(source_audio_files)
        assert (len(source_audio_files) > 0)
        sr, waveform = wavfile.read(source_audio_files[0])
        waveform = waveform.astype(np.float)
        is_all_zero = np.all((waveform == 0))
        if not is_all_zero:
            waveform = waveform / (2 ** 15)
        waveform = waveform.T.copy()
        source_audios.append(waveform)

        source_azi_angle = metadata[key]['panning_angles'][0]
        source_zen_angle = metadata[key]['panning_angles'][1]
        source_positions.append([source_azi_angle, source_zen_angle])

    mix_path = os.path.join(curr_dir, 'mix.wav')
    sr, mix_waveform = wavfile.read(mix_path)
    mix_waveform = mix_waveform.astype(np.float)
    mix_is_all_zero = np.all((mix_waveform == 0))
    if not mix_is_all_zero:
        mix_waveform = mix_waveform / np.amax(np.abs(mix_waveform[:, 0])) / np.sqrt(2 * ambiorder + 1)

    return mix_waveform, source_positions, source_audios, sorted(metadata.keys())


def main(evaluate_dir, model_checkpoint, result_dir):
    ambiorder = 4
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    print('Start evaluating!')

    n_channels = (ambiorder + 1) ** 2
    model = TungYu()
    model.load_state_dict(torch.load(model_checkpoint), strict=True)
    model.train = False
    model = model.to(device)

    all_dirs = sorted(list(Path(evaluate_dir).glob('[0-9]*')))
    # print(all_dirs)
    si_sdr_nn = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_beamformer_max_di = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_beamformer_max_re = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_beamformer_max_sdr = {'vocals': [], 'drums': [], 'bass': []}
    si_sdr_omnimix = {'vocals': [], 'drums': [],
                      'bass': []}  # For baseline, we consider the omni mix as the separated source

    si_sdr_stats = {'median': {'nn': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'beamformer_max_di': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'beamformer_max_re': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'beamformer_max_sdr': {'vocals': None, 'drums': None, 'bass': None, 'all': None},
                               'omnimix': {'vocals': None, 'drums': None, 'bass': None, 'all': None}}}

    for idx in range(0, len(all_dirs)):
        print(idx)
        curr_dir = all_dirs[idx]
        get_waveform_n_angle(curr_dir)
        # print(curr_dir)  # /home/tungyu/Project/musdb18/different_angle/15/00000
        mixed_data, source_positions, source_audios, source_name = get_waveform_n_angle(curr_dir)
        mixed_data = mixed_data[:, 0:n_channels]
        for [azi_angle, zen_angle], target_waveform, key in zip(source_positions, source_audios, source_name):
            azi_angle_beamformer = azi_to_0_2pi_range(azi_angle)
            ele_angle_beamformer = zen_to_ele(zen_angle)
            # print(azi_angle_beamformer, ele_angle_beamformer, key)
            audio_bf_max_re = beamformer_max_re(mixed_data, np.array((azi_angle_beamformer, ele_angle_beamformer)))
            # print(audio_bf_max_re.shape)
            wavfile.write(str(curr_dir) + '/' + str(key) + '_bf_max_re.wav', 44100, audio_bf_max_re)

            # nn output here!
            # audio_bf_max_re = audio_bf_max_re / (2 ** 15)
            # print(audio_bf_max_re)
            nn_mixed_data = mixed_data[:, 0:4]
            nn_mixed_data = torch.tensor(nn_mixed_data.T)
            nn_bf_data = torch.tensor(audio_bf_max_re)
            nn_predicted_data = forward_pass(model, nn_mixed_data, nn_bf_data, device=device)
            nn_predicted_audio = nn_predicted_data.detach().cpu().numpy()
            wavfile.write(str(curr_dir) + '/' + str(key) + '_predicted.wav', 44100, nn_predicted_audio)

            is_all_zero = np.all((target_waveform == 0))
            if not is_all_zero:
                target_waveform = target_waveform
                nn_predicted_audio = nn_predicted_audio.flatten()
                nn_bf_data = nn_bf_data.detach().cpu().numpy().flatten()

                si_sdr_pt = si_sdr(nn_predicted_audio, target_waveform)
                si_sdr_nn[key].append(si_sdr_pt)
                si_sdr_at = si_sdr(nn_bf_data, target_waveform)
                si_sdr_beamformer_max_re[key].append(si_sdr_at)

    json_path = os.path.join(result_dir, 'si_sdr_nn.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_nn, fp)

    json_path = os.path.join(result_dir, 'si_sdr_beamformer_max_re.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_beamformer_max_re, fp)

    for key in ['vocals', 'drums', 'bass']:
        si_sdr_stats['median']['nn'][key] = stat.median(si_sdr_nn[key])
        si_sdr_stats['median']['beamformer_max_di'][key] = stat.median(si_sdr_beamformer_max_di[key])
        si_sdr_stats['median']['beamformer_max_re'][key] = stat.median(si_sdr_beamformer_max_re[key])
        si_sdr_stats['median']['beamformer_max_sdr'][key] = stat.median(si_sdr_beamformer_max_sdr[key])
        si_sdr_stats['median']['omnimix'][key] = stat.median(si_sdr_omnimix[key])

    si_sdr_stats['median']['nn']['all'] = stat.median(
        flatten([si_sdr_nn['vocals'], si_sdr_nn['drums'], si_sdr_nn['bass']]))
    si_sdr_stats['median']['beamformer_max_di']['all'] = stat.median(flatten(
        [si_sdr_beamformer_max_di['vocals'], si_sdr_beamformer_max_di['drums'], si_sdr_beamformer_max_di['bass']]))
    si_sdr_stats['median']['beamformer_max_re']['all'] = stat.median(flatten(
        [si_sdr_beamformer_max_re['vocals'], si_sdr_beamformer_max_re['drums'], si_sdr_beamformer_max_re['bass']]))
    si_sdr_stats['median']['beamformer_max_sdr']['all'] = stat.median(flatten(
        [si_sdr_beamformer_max_sdr['vocals'], si_sdr_beamformer_max_sdr['drums'], si_sdr_beamformer_max_sdr['bass']]))
    si_sdr_stats['median']['omnimix']['all'] = stat.median(
        flatten([si_sdr_omnimix['vocals'], si_sdr_omnimix['drums'], si_sdr_omnimix['bass']]))

    json_path = os.path.join(result_dir, 'si_sdr_stats.json')
    with open(json_path, 'w') as fp:
        json.dump(si_sdr_stats, fp)


if __name__ == '__main__':
    main(evaluate_dir='/home/tungyu/Project/musdb18/different_angle/90',
         model_checkpoint='/home/tungyu/Project/STFTNet/best.pt',
         result_dir='/home/tungyu/Project/musdb18/different_angle/90')
