import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import torch

""" Coordinates """


def sph2cart(azimuth, zenith):  # 球坐标转为笛卡尔坐标
    x = np.cos(np.pi / 2 - zenith) * np.cos(azimuth)
    y = np.cos(np.pi / 2 - zenith) * np.sin(azimuth)
    z = np.sin(np.pi / 2 - zenith)

    cart = np.array([x, y, z])

    return cart


def cart2sph(c):
    azi = np.arctan2(c[1], c[0])
    zen = np.pi / 2 - np.arctan2(c[2], np.sqrt(c[0] ** 2 + c[1] ** 2))
    return np.array([azi, zen])


def great_circle_distance(azi0, zen0, azi1, zen1):
    c1 = sph2cart(azi0, zen0)
    c2 = sph2cart(azi1, zen1)

    c1 = c1 / np.sqrt(np.sum(c1 ** 2))
    c2 = c2 / np.sqrt(np.sum(c2 ** 2))

    if (c1 == c2).all():
        return 0

    else:
        phi = np.arccos(np.inner(c1, c2))
        return phi


def azi_to_0_2pi_range(azi_angle):
    if azi_angle < 0:
        return azi_angle + 2 * np.pi
    else:
        return azi_angle


def zen_to_ele(zen_angle):
    return np.pi / 2 - zen_angle


""" Ambisonics """


def eval_sh(max_order, dirs_sph):
    # eval_sh ... evaluate spherical harmonics up to maximal order max_order
    # inputs:      max_order ... maximal SH order
    #              dirs_sph ... DoA a (Q, 2) array with (azimuth, elevation) in rad #
    #                           in case of one direction, also (2,) is possible
    # outputs:     Y ... (Q, (max_order + 1)^2) matrix of real spherical harmonics

    dirs_sph = np.array(dirs_sph)

    if dirs_sph.ndim == 1:
        dirs_sph = dirs_sph.reshape((1, 2))

    num_sh_channels = (max_order + 1) ** 2
    num_dir = dirs_sph.shape[0]

    azi = dirs_sph[:, 0]
    ele = dirs_sph[:, 1]

    Y = np.zeros((num_dir, num_sh_channels))

    for n in range(0, max_order + 1):
        for m in range(-n, n + 1):

            # Y_n^m(/theta, /phi) = sqrt((2n+1)(n-m)!/4pi(n+m)!) * P_n^m * cos(/theta) * exp(im/phi)

            i = n + n ** 2 + m

            if m < 0:
                Yazi = np.array(np.sqrt(2) * np.sin(azi * np.abs(m)))
            elif m > 0:
                Yazi = np.array(np.sqrt(2) * np.cos(azi * m))
            else:
                Yazi = np.ones(num_dir)

            Yzen = np.zeros(num_dir)
            for iDir in range(0, num_dir):
                # scipy.special.lpmv()用于计算 P_n^m 的连带勒让德多项式的值
                Yzen[iDir] = (-1) ** m * scipy.special.lpmv(np.abs(m), n, np.cos(np.pi / 2 - ele[iDir]))

            normlz = np.sqrt(
                (2 * n + 1) * np.math.factorial(n - np.abs(m)) / (4 * np.pi * np.math.factorial(n + np.abs(m))))
            # np.math.factorial: 阶乘

            Y[:, i] = Yazi * Yzen * normlz

    return Y


def beamformer_max_di(input_signal, dir_sph):
    ambi_order = int(np.sqrt(input_signal.shape[1]) - 1)
    y = eval_sh(ambi_order, dir_sph)
    output_signal = np.dot(input_signal, np.transpose(y))  # s_max_di = input_signal · SH_signal^T
    return output_signal


def beamformer_max_re(input_signal, dir_sph):  # max-re算法进行波束形成
    ambi_order = int(np.sqrt(input_signal.shape[1]) - 1)

    weights_per_sh_channel = expand_weights(getMaxReWeights(ambi_order))

    y = eval_sh(ambi_order, dir_sph)

    # print("input_signal sh max: ", np.max(y))

    output_signal = np.dot(np.multiply(input_signal, weights_per_sh_channel), np.transpose(y))

    # print("input_signal bf max: ", np.max(output_signal / 2 ** 15))

    return output_signal


def beamformer_max_sdr(input_signal, gt_signal):
    C = input_signal.T @ input_signal  # @ 用于计算矩阵乘法
    try:
        # 根据公式 d_{max-SDR} = C^(-1) X_N^T s
        # 使用 np.linalg.solve求解线性方程，其中 C 是系数矩阵，input_signal.T @ gt_signal 是常向量。从而计算得到 d_{max-SDR}
        weights = np.linalg.solve(C, input_signal.T @ gt_signal)
        singular_matrix = False  # 奇异矩阵
        output_signal = input_signal @ weights
        return output_signal, singular_matrix

    except np.linalg.LinAlgError as err:  # 线性方程无法求解
        if 'Singular matrix' in str(err):
            singular_matrix = True
            return None, singular_matrix


def expand_weights(weights_per_order):  # 直接用就行
    ambi_order = weights_per_order.shape[0] - 1
    num_sh_channels = (ambi_order + 1) ** 2

    weights_per_sh_channel = np.zeros(num_sh_channels)

    i = 0
    for n in range(0, ambi_order + 1):
        num_channels = 2 * n + 1
        weights_per_sh_channel[i:(i + num_channels)] = weights_per_order[n]
        i += num_channels

    return weights_per_sh_channel


def getMaxReWeights(order, maxOrder=None):  # 直接用就行
    # Returns maxRe weights for a given order
    # if additionally, as maxOrder is specified,
    # zeros are padded to the coefficients

    if maxOrder == None:
        maxOrder = order

    maxReWeights = [np.array([1]), np.array([0.3659487, 0.2113504]), np.array([0.18752139, 0.14536802, 0.07527491]),
                    np.array([0.11367906, 0.09794028, 0.06973125, 0.03483484]),
                    np.array([0.0761984, 0.0690694, 0.0558118, 0.03827066, 0.01884882])]

    if order == maxOrder:
        return maxReWeights[order] * 1.0 / 0.079
    elif order <= maxOrder:
        return np.hstack((maxReWeights[order], np.zeros(maxOrder - order))) * 1.0 / 0.079
    elif order > maxOrder:
        return maxReWeights[maxOrder] * 1.0 / 0.079


""" Evaluation """


def si_sdr(estimated_signal, reference_signals, scaling=True, eps=1e-8):
    # 计算 SI-SDR 评估分离效果

    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / (Rss + eps)
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true ** 2).sum()
    Snn = (e_res ** 2).sum() + eps

    SDR = 10 * math.log10((Sss / Snn) + eps)

    return SDR


def si_sdr_torch_edition(estimated_signal, reference_signals, scaling=True, eps=1e-8):
    Rss = torch.sum(reference_signals * reference_signals, dim=-1, keepdim=True)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = torch.sum(this_s * estimated_signal, dim=-1, keepdim=True) / (Rss + eps)
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = torch.sum(e_true ** 2, dim=-1)
    Snn = torch.sum(e_res ** 2, dim=-1) + eps

    SDR = torch.log10((Sss / Snn) + eps)

    return -SDR


def negative_si_snr(estimated_signal, reference_signals, eps=1e-13):
    def l2norm(mat, keep_dim=False):
        return torch.norm(mat, dim=1, keep_dim=keep_dim)

    t = torch.sum(estimated_signal * reference_signals, dim=-1, keepdim=True) * reference_signals / (
            l2norm(reference_signals, keep_dim=True) ** 2 + eps)
    return -torch.mean(torch.log10(eps + l2norm(t) / (l2norm(estimated_signal - t) + eps)))


def frequency_mse(estimated_signal, reference_signals):
    n_fft = 512
    window_length = 512
    hop_length = 256
    window_function = torch.hann_window(window_length=window_length).to('cuda:0')
    b_s, n_c, s_l = estimated_signal.shape
    estimated_signal = estimated_signal.view(b_s * n_c, s_l)
    reference_signals = reference_signals.view(b_s * n_c, s_l)
    estimated_signal_mag = torch.stft(estimated_signal, n_fft, hop_length=hop_length, win_length=window_length,
                                      window=window_function, center=True, return_complex=True)
    reference_signals_mag = torch.stft(reference_signals, n_fft, hop_length=hop_length, win_length=window_length,
                                       window=window_function, center=True, return_complex=True)

    fre_mes_loss = torch.mean((torch.abs(estimated_signal_mag) - torch.abs(reference_signals_mag)) ** 2)
    return fre_mes_loss


""" Plotting """


def projectHammerAitof(azi, ele):
    azi = np.mod(azi + np.pi, 2 * np.pi) - np.pi

    x = -np.multiply(np.cos(ele), np.sin(azi / 2))
    y = 0.5 * np.sin(ele)

    normalization = np.sqrt(1 + np.multiply(np.cos(ele), np.cos(azi / 2)))

    x = np.divide(x, normalization)
    y = np.divide(y, normalization)

    return x, y


def drawGrid():
    aziGrid = np.array([0, -30, 30, -60, 60, -90, 90, -120, 120, -150, 150, 179.9]) / 180 * np.pi
    eleGrid = np.array([-60, -45, -15, -30, 0, 15, 30, 45, 60]) / 180 * np.pi

    for aziAngle in aziGrid:
        ele = np.linspace(-np.pi / 2, np.pi / 2, 100)
        azi = np.ones(100) * aziAngle
        x, y = projectHammerAitof(azi, ele)

        plt.text(x[50], y[50], str(int(np.rint(aziAngle * 180 / np.pi))), fontsize=10, color='grey', alpha=1)

        plt.plot(x, y, '--', color='grey', alpha=0.7)

    for eleAngle in eleGrid:
        azi = np.linspace(-np.pi, np.pi - 0.001, 100)
        ele = np.ones(100) * eleAngle
        x, y = projectHammerAitof(azi, ele)

        if eleAngle != 0:
            plt.text(x[50], y[50], str(int(np.rint(eleAngle * 180 / np.pi))), fontsize=10, color='grey', alpha=1)

        plt.plot(x, y, '--', color='grey', alpha=0.7)
    plt.axis('off')

