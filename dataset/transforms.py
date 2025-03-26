import numpy as np
import torch
import librosa
import random
import config
import scipy

random.seed(42)


# Composes several transforms together.
class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# Default data augmentation
class Padding:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, sound):
        return np.pad(sound, self.pad, "constant")


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, sound):
        org_size = len(sound)
        start = random.randint(0, org_size - self.size)
        return sound[start : start + self.size]


class Normalize:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sound):
        return sound / self.factor


class RandomScale:
    def __init__(self, max_scale, interpolate="Linear"):
        self.max_scale = max_scale
        self.interpolate = interpolate

    def __call__(self, sound):
        scale = np.power(self.max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if self.interpolate == "Linear":
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif self.interpolate == "Nearest":
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception("Invalid interpolation mode {}".format(self.interpolate))

        return scaled_sound


class RandomGain:
    def __init__(self, db):
        self.db = db

    def __call__(self, sound):
        return sound * np.power(10, random.uniform(-self.db, self.db) / 20.0)


class MultiCrop:
    def __init__(self, input_length, n_crops):
        self.input_length = input_length
        self.n_crops = n_crops

    def __call__(self, sound):
        stride = (len(sound) - self.input_length) // (self.n_crops - 1)
        sounds = [
            sound[stride * i : stride * i + self.input_length]
            for i in range(self.n_crops)
        ]
        return np.array(sounds)


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (
        2 * np.log10(12194)
        + 2 * np.log10(freq_sq)
        - np.log10(freq_sq + 12194**2)
        - np.log10(freq_sq + 20.6**2)
        - 0.5 * np.log10(freq_sq + 107.7**2)
        - 0.5 * np.log10(freq_sq + 737.9**2)
    )
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode="A_weighting"):
    if fs == 16000 or fs == 20000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception("Invalid fs {}".format(fs))
    stride = n_fft // 2

    gain = []
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == "RMSE":
            g = np.mean(sound[i : i + n_fft] ** 2)
        elif mode == "A_weighting":
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i : i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception("Invalid mode {}".format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


"""def get_A_weighting(fs, n_fft):
    freqs = np.linspace(0, fs / 2, n_fft // 2 + 1)
    return librosa.A_weighting(freqs)


def compute_gain(sound, fs, A_weighting, n_fft=2048, hop_length=1024):
    # Compute the STFT
    stft = librosa.stft(sound, n_fft=n_fft, hop_length=hop_length)

    # Convert to power spectral density
    psd = np.abs(stft) ** 2

    # Apply the A-weighting filter to the PSD
    weighted_psd = psd * np.power(10, A_weighting / 10.0)[:, np.newaxis]

    # Convert to decibels
    weighted_psd_db = 10.0 * np.log10(weighted_psd + 1e-10)

    # Calculate the gain as the maximum of the weighted PSD in dB
    gain = np.max(weighted_psd_db, axis=0)

    return gain"""


def mix(sound1, sound2, r, fs):
    n_fft = 2048
    hop_length = n_fft // 2
    # A_weighting = get_A_weighting(fs, n_fft)

    # Compute gains
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))

    # Calculate mixing ratio
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.0) * (1 - r) / r)

    # Mix sounds
    sound = (sound1 * t + sound2 * (1 - t)) / np.sqrt(t**2 + (1 - t) ** 2)

    return sound


# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = "{}h{:02d}m".format(h, m)
    else:
        line = "{}m{:02d}s".format(m, s)

    return line