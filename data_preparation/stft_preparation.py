import numpy as np
import librosa
import soundfile as sf
import h5py

# Set parameters
frame_size = 2048  # 43 ms frame
hop_size = frame_size // 2  # 50% overlap
segment_duration = 0.17  # 170 ms
sr = 48000  # Sampling rate
num_mics = 4  # Four microphones

def process_audio(file_path):
    # Load audio using soundfile to preserve multi-channel
    audio, sr = sf.read(file_path)
    print(sr)

    # Ensure correct shape (channels, samples)
    if audio.ndim == 1:
        raise ValueError("The audio file has only one channel. Expected 4 channels.")
    audio = audio.T  # Transpose to (channels, samples)

    if audio.shape[0] != num_mics:
        raise ValueError(f"Expected {num_mics} channels, but got {audio.shape[0]}.")

    # Calculate segment size (170ms in samples)
    segment_samples = int(segment_duration * sr)
    total_samples = audio.shape[1]

    # Compute number of full segments
    num_segments = total_samples // segment_samples

    # Get frequency bins
    freqs = np.fft.rfftfreq(frame_size, d=1/sr)
    valid_bins = np.where((freqs >= 100) & (freqs <= 8000))[0]
    print(valid_bins.shape)

    # Storage for all segments
    output_data = np.zeros((num_segments, num_mics * 2, 337, 7), dtype=np.float32)

    # Process each segment separately
    for i in range(num_segments):
        segment = audio[:, i * segment_samples:(i + 1) * segment_samples]

        # Compute STFT for each microphone
        stfts = np.array([librosa.stft(segment[ch], n_fft=frame_size, hop_length=hop_size) for ch in range(num_mics)])

        # Keep only the desired frequency bins
        stfts = stfts[:, valid_bins, :7]  # Ensure exactly 7 time frames
        print(stfts[0,0,0])

        # Separate real and imaginary parts
        real_part = np.real(stfts)
        imag_part = np.imag(stfts)

        # Interleave real and imaginary parts into 8 channels
        output_data[i, 0::2] = real_part  # Even indices store real parts
        output_data[i, 1::2] = imag_part  # Odd indices store imaginary parts

    return output_data

def save_to_hdf5(data, output_file):
    with h5py.File(output_file, "w") as f:
        f.create_dataset("stft", data=data)

# Example usage
file_path = r"C:\Users\grizi\Downloads\sslr\sslr\sslr\lsp_test_106\audio\ssl-data_2017-04-29-16-56-04_3.wav"
# file_path = r"C:\Users\grizi\Desktop\TUD\year2\thesis\neural_network\DoA_Net\data\ov1_mic_dev\train\fold2_room1_mix001_ov1.wav"

output_data = process_audio(file_path)
# save_to_hdf5(output_data, "processed_audio.h5")

print("Processing complete. Data shape:", output_data.shape)
