# for signal processing
sample_rate = 32000 # 16000 for scv2, 32000 for audioset and esc-50
clip_samples = sample_rate * 5 # audio_set 10-sec clip
mel_bins = 64
window_size = 1024
hop_size = 800
fmin = 50
fmax = 14000
shift_max = int(clip_samples * 0.5)
audio_duration = 5

# data preprocess
inputLength = 32000 * 5