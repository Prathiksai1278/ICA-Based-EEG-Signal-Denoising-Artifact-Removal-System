import mne
import numpy as np
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch

file_path = "S001R01.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)

new_names = {}
for ch in raw.ch_names:
    clean_name = ch.replace('.', '').upper()
    new_names[ch] = clean_name
raw.rename_channels(new_names)

print("\nShowing RAW EEG")
raw.plot(duration=5, n_channels=10, block=True)

raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)
raw_notch = raw_filtered.copy().notch_filter(freqs=50)


ica = ICA(n_components=20, random_state=42, max_iter=800)
ica.fit(raw_notch)

sources = ica.get_sources(raw_notch).get_data()
sfreq = raw.info['sfreq']

print("\n==============================")
print("Improved ICA Component Analysis")
print("==============================")

feature_list = []

for i in range(len(sources)):
    component = sources[i]

    variance = np.var(component)
    psd, freqs = psd_array_welch(component, sfreq=sfreq)

    delta_power = np.mean(psd[(freqs >= 1) & (freqs < 4)])
    gamma_power = np.mean(psd[(freqs > 30)])

    feature_list.append((i, variance, delta_power, gamma_power))


feature_array = np.array(feature_list, dtype=object)

variances = feature_array[:,1].astype(float)
delta_powers = feature_array[:,2].astype(float)
gamma_powers = feature_array[:,3].astype(float)


var_threshold = np.mean(variances) + 2*np.std(variances)
delta_threshold = np.mean(delta_powers) + 2*np.std(delta_powers)
gamma_threshold = np.mean(gamma_powers) + 2*np.std(gamma_powers)

bad_components = []

for i, var, delta, gamma in feature_list:
    if (var > var_threshold) or \
       (delta > delta_threshold) or \
       (gamma > gamma_threshold):
        bad_components.append(i)


max_remove = 10
bad_components = bad_components[:max_remove]

print("Components selected for removal:", bad_components)

print("\nShowing ICA Sources")
ica.plot_sources(raw_notch, block=True)


ica.exclude = bad_components
raw_clean = raw_notch.copy()
ica.apply(raw_clean)

print("\nShowing CLEANED EEG")
raw_clean.plot(duration=5, n_channels=10, block=True)

print("Controlled Artifact Removal Completed")


raw_data = raw_notch.get_data()
clean_data = raw_clean.get_data()

signal_power = np.mean(clean_data ** 2)
noise = raw_data - clean_data
noise_power = np.mean(noise ** 2)

snr = 10 * np.log10(signal_power / noise_power)

print("\n===================================")
print(f"SNR after controlled cleaning: {snr:.2f} dB")
print("===================================")
