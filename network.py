import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResidualBlock
from utils import si_sdr_torch_edition
import torchaudio.transforms as T


def load_pretrain(model, state_dict):
    for key in state_dict.keys():
        try:
            _ = model.load_state_dick({key: state_dict[key]}, strict=False)
            print("loading pretrained model")
        except Exception as e:
            print("fffff")
            print(e)


class TungYu(nn.Module):
    def __init__(self,
                n_audio_channels: int = 5,
                ambimode: str = 'mixed',
                conditioning_size: int = 2,
                padding: int = 8,
                context: int = 3,
                depth: int = 6,
                channels: int = 32,  # original paper is 64
                growth: float = 2.0,
                lstm_layers: int = 2,
                rescale: float = 0.1,
                stft_window_size: int = 512,
                stft_hop_length: int = 256,
                activation: str = 'relu'
                ):
        super(TungYu, self).__init__()
        self.n_audio_channels = n_audio_channels
        self.conditioning_size = conditioning_size
        self.padding = padding
        self.context = context
        self.depth = depth
        self.channels = channels  # 32
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale
        self.ambimode = ambimode
        self.stft_window_size = stft_window_size
        self.stft_hop_length = stft_hop_length
        self.activation = activation

        in_channels = n_audio_channels  # 5

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=context, padding=padding)  # 5, 32, 3, 8
        self.res_block1 = ResidualBlock(channels, channels)
        self.res_block2 = ResidualBlock(2 * channels, 2 * channels)
        self.res_block3 = ResidualBlock(4 * channels, 4 * channels)
        self.relu = nn.ReLU()


        channels = 32 * channels

        self.lstm = nn.LSTM(bidirectional=False, num_layers=lstm_layers, hidden_size=channels,
                            input_size=channels)
        self.lstm_linear = nn.Linear(2 * channels, channels)

        self.mask_linear = nn.Linear(channels, 257)

    def forward(self, mix, beamformer_audio):
        print(mix.shape)
        print(beamformer_audio.shape)
        # input shape: (batch_size, 5, 257, 1206)
        n_fft = 512
        window_length = 512
        hop_length = 256
        sr = 44100
        window_function = torch.hann_window(window_length=window_length).to('cuda:0')
        mix = torch.cat((mix, beamformer_audio), dim=1)
        # print("look: ", mix.shape)
        b_s, n_c, s_l = mix.shape
        mix_reshaped = mix.view(b_s * n_c, s_l)

        mix_mag = torch.stft(mix_reshaped, n_fft, hop_length=hop_length, win_length=window_length, window=window_function,
                             center=True, return_complex=True)
        mix_mag = mix_mag.view(b_s, n_c, mix_mag.shape[1], mix_mag.shape[2])
        mix_mag_real = torch.abs(mix_mag)

        x = mix_mag_real

        x = self.conv1(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.relu(x)

        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)
        # x = self.lstm_linear(x)

        mask = torch.sigmoid(self.mask_linear(x))
        mask = mask.permute(0, 2, 1)

        separated_spec = mix_mag[4] * mask
        separated_spec = separated_spec.view(-1, 257, 1206)

        separated_audio = torch.istft(separated_spec, n_fft=n_fft, hop_length=hop_length, win_length=window_length,
                                      center=True)

        return separated_audio

    def loss(self, output_signals, gt_output_signals):
        si_sdr_loss = 0.0
        batch_size = output_signals.size(0)
        for i in range(batch_size):
            estimated_signal = output_signals[i, :, :]
            reference_signal = gt_output_signals[i, :, :]
            si_sdr_loss -= si_sdr_torch_edition(estimated_signal, reference_signal)

        si_sdr_loss = si_sdr_loss / batch_size
        loss = F.l1_loss(output_signals, gt_output_signals) + si_sdr_loss * 1.3
        return loss
