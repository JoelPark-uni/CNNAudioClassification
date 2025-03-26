# for signal processing
sample_rate = 32000 # 16000 for scv2, 32000 for audioset and esc-50
mel_bins = 64
window_size = 1024
hop_size = 512
fmin = 50
fmax = 14000
audio_duration = 5
clip_samples = sample_rate * audio_duration # audio_set 10-sec clip
shift_max = int(clip_samples * 0.5)

# data preprocess
inputLength = sample_rate * audio_duration