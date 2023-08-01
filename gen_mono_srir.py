"""
--------------------------------------
Generate mono-audio srir simulation signal
Created by TungYu Dominick Yeung 01.08.2023
srir1 -> vocals
srir2 -> drums
srir3 -> bass
--------------------------------------
"""
import scipy.io.wavfile
import numpy as np
import scipy as sci


def gen_mono_srir_audio(mono_audio, srir_sigal):
    sr, mono_signal = scipy.io.wavfile.read(mono_audio)
    _, srir_data = scipy.io.wavfile.read(srir_sigal)
    length = len(mono_audio)
    num_sh_channels = 25
    mono_srir_audios_ambi = np.zeros((length + srir_data.shape[0] - 1, num_sh_channels))
    for channel in range(num_sh_channels):
        mono_srir_audios_ambi[:, channel] = sci.signal.convolve(mono_signal, srir_data[:, channel])

    mono_srir_audios = mono_srir_audios_ambi
    return mono_srir_audios


if __name__ == "__main__":
    mono_audio_path = '../musdb18/mini_dataset/train/00000/vocals.wav'
    srir_signal_path = '../musdb18/mini_dataset/train/00000/srir1.wav'
    sampling_rate = 44100

    mono_srir_audio = gen_mono_srir_audio(mono_audio_path, srir_signal_path)
    scipy.io.wavfile.write('../musdb18/mini_dataset/train/00000/vocals_single_srir.wav', sampling_rate,
                           (mono_srir_audio * np.iinfo(np.int16).max).astype(np.int16))
