import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import ResidualBlock
from utils import si_sdr_torch_edition, frequency_mse

def load_pretrain(model, state_dict):
    for key in state_dict.keys():
        try:
            _ = model.load_state_dick({key: state_dict[key]}, strict=False)
            print("loading pretrained model")
        except Exception as e:
            print("fffff")
            print(e)


def adjust_length(beamformer_sigal, separated_signal):
    target_length = beamformer_sigal.size(-1)
    true_length = separated_signal.size(-1)
    if true_length > target_length:
        separated_signal = separated_signal[:, :target_length]
    elif true_length < target_length:
        separated_signal = F.pad(separated_signal, (0, target_length - true_length))
    else:
        pass
    return separated_signal


class TungYu(nn.Module):
    def __init__(self,
                 conditioning_size: int = 2,
                 padding: int = 8,
                 context: int = 3,
                 depth: int = 6,
                 channels: int = 32,  # original paper is 64
                 growth: float = 2.0,
                 lstm_layers: int = 1,
                 stft_window_size: int = 512,
                 stft_hop_length: int = 256,
                 activation: str = 'relu'
                 ):
        super(TungYu, self).__init__()
        self.conditioning_size = conditioning_size
        self.padding = padding
        self.context = context
        self.depth = depth
        self.channels = 32  # 32
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.stft_window_size = stft_window_size
        self.stft_hop_length = stft_hop_length
        self.activation = activation

        in_channels = 5

        self.conv1 = nn.Conv2d(in_channels, self.channels, kernel_size=context, padding=context // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.maxpool1 = nn.MaxPool2d((2, 1))
        self.res_block1 = ResidualBlock(self.channels, 2 * self.channels)
        self.res_block2 = ResidualBlock(2 * self.channels, 4 * self.channels)
        self.res_block3 = ResidualBlock(4 * self.channels, 4 * self.channels)
        self.relu = nn.ReLU()

        channels = 4 * channels

        self.linear = nn.Linear(channels * 64, 128)  # original: nn.Linear(channels * 64, 128)

        self.lstm = nn.LSTM(bidirectional=False, num_layers=lstm_layers, hidden_size=channels,
                            input_size=channels)

        self.mask_linear = nn.Linear(channels, 257)

    def forward(self, mix, beamformer_audio):
        mix = mix.to('cuda:0')
        beamformer_audio = beamformer_audio.to('cuda:0')
        n_fft = 512
        window_length = 512
        hop_length = 256
        window_function = torch.hann_window(window_length=window_length).to('cuda:0')
        mix = torch.cat((mix, beamformer_audio), dim=1)

        b_s, n_c, s_l = mix.shape
        mix_reshaped = mix.view(b_s * n_c, s_l)

        mix_mag = torch.stft(mix_reshaped, n_fft, hop_length=hop_length, win_length=window_length,
                             window=window_function, center=True, return_complex=True)

        mix_mag = mix_mag.view(b_s, n_c, mix_mag.shape[1], mix_mag.shape[2])  # torch.Size([batch_size, 5, 257, 1206])
        mix_mag_real = torch.abs(mix_mag)

        x = mix_mag_real
        x = self.conv1(x)
        x = self.bn1(x)  # batch normalization
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.res_block1(x)
        x = self.maxpool1(x)
        x = self.res_block2(x)

        x = x.permute(0, 3, 1, 2)  # shape: torch.Size([2, 512, 309942])
        x = x.view(x.size(0), x.size(1), -1)

        x = self.linear(x)
        x, _ = self.lstm(x)

        mask = torch.sigmoid(self.mask_linear(x))
        mask = mask.permute(0, 2, 1)

        separated_spec = mix_mag[:, 4] * mask

        separated_audio = torch.istft(separated_spec, n_fft=n_fft, hop_length=hop_length, win_length=window_length,
                                      center=True)

        separated_audio = adjust_length(beamformer_audio, separated_audio)

        separated_audio = separated_audio.unsqueeze(1)
        return separated_audio, mask

    def L1_loss(self, output_signals, gt_output_signals):
        return F.l1_loss(output_signals, gt_output_signals)

    def loss(self, output_signals, gt_output_signals):
        si_sdr_loss_ = []
        si_sdr_loss = 0.0
        batch_size = output_signals.size(0)
        for i in range(batch_size):
            estimated_signal = output_signals[i, :, :]
            reference_signal = gt_output_signals[i, :, :]
            si_sdr_i = si_sdr_torch_edition(estimated_signal, reference_signal)
            si_sdr_loss_.append((si_sdr_i).detach().cpu().numpy())
            si_sdr_loss += si_sdr_i
        si_sdr_loss = si_sdr_loss / batch_size
        mse_loss = frequency_mse(output_signals, gt_output_signals)
        loss = si_sdr_loss + mse_loss
        return loss, mse_loss, si_sdr_loss_
