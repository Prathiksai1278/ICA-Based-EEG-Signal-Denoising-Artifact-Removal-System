import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch


file_path = "S001R01.edf"
raw = mne.io.read_raw_edf(file_path, preload=True)


new_names = {}
for ch in raw.ch_names:
    clean_name = ch.replace('.', '').upper()
    new_names[ch] = clean_name
raw.rename_channels(new_names)


raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=40.0)
raw_notch = raw_filtered.copy().notch_filter(freqs=50)


ica = ICA(n_components=20, random_state=42, max_iter=800)
ica.fit(raw_notch)

sources = ica.get_sources(raw_notch).get_data()
sfreq = raw.info['sfreq']

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


max_remove = 4
bad_components = bad_components[:max_remove]

print("Components selected for removal:", bad_components)


ica.exclude = bad_components
raw_clean = raw_notch.copy()
ica.apply(raw_clean)

print("Controlled Artifact Removal Completed")


channel_index = 10
raw_data = raw_notch.get_data()[channel_index]
clean_data = raw_clean.get_data()[channel_index]

time = np.arange(len(raw_data)) / sfreq


plt.figure(figsize=(12,6))
plt.plot(time, raw_data, label="Raw Signal", alpha=0.7)
plt.plot(time, clean_data, label="Cleaned Signal", alpha=0.7)
plt.title("Before vs After Cleaning (Overlay)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

noise_removed = raw_data - clean_data
plt.figure(figsize=(12,4))
plt.plot(time, noise_removed)
plt.title("Removed Noise Component")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()

psd_raw, freqs = psd_array_welch(raw_data, sfreq=sfreq)
psd_clean, _ = psd_array_welch(clean_data, sfreq=sfreq)

plt.figure(figsize=(10,6))
plt.semilogy(freqs, psd_raw, label="Raw")
plt.semilogy(freqs, psd_clean, label="Cleaned")
plt.title("Power Spectrum Comparison")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.legend()
plt.show()

energy_raw = np.mean(raw_data**2)
energy_clean = np.mean(clean_data**2)
energy_noise = np.mean(noise_removed**2)

plt.figure(figsize=(6,5))
plt.bar(["Raw Energy", "Clean Energy", "Removed Noise"],
        [energy_raw, energy_clean, energy_noise])
plt.title("Energy Comparison")
plt.ylabel("Mean Power")
plt.show()


signal_power = energy_clean
noise_power = energy_noise
snr = 10 * np.log10(signal_power / noise_power)

print("\n===================================")
print(f"SNR after controlled cleaning: {snr:.2f} dB")
print("===================================")
