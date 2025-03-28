import h5py
import numpy as np
import librosa

# Path to the H5 file
h5_file_path = r"C:\Users\grizi\Desktop\TUD\year2\thesis\neural_network\dataset_tools\car_test_dataset_2.h5"

# Open the H5 file
with h5py.File(h5_file_path, 'r') as h5file:
    # Iterate through each dataset in the H5 file
    for dataset_name in h5file:
        # Get the dataset
        dataset = h5file[dataset_name]
        if dataset_name == 'audio':
            audio_data = np.array(dataset)
        elif dataset_name == 'label':
            label_data = np.array(dataset)
        elif dataset_name == 'sound_data':
            audio_data = np.array(dataset)
        
        # Print the dataset name and its shape
        print(f"Dataset: {dataset_name}, Shape: {dataset.shape}")

def process_audio(audio):

    # set the parameter for stft processing 
    sr =  48000
    num_mics = 32  # Four microphones
    num_segments = len(audio)
    frame_size = 512  # time window
    hop_size = frame_size // 2  # 50% overlap

    # Get frequency bins
    freqs = np.fft.rfftfreq(frame_size, d=1/sr)
    valid_bins = np.where((freqs >= 100) & (freqs <= 8000))[0]
    print(valid_bins)

    # Storage for all segments
    output_data = np.zeros((num_segments, num_mics * 2, 84, 15), dtype=np.float32)

    # Process each segment separately
    for i in range(len(audio)):
        segment = audio[i,:].T
        print(segment.shape,frame_size,hop_size)

        # Compute STFT for each microphone
        stfts = np.array([librosa.stft(segment[ch].astype(np.float32), n_fft=frame_size, hop_length=hop_size) for ch in range(num_mics)])
        print(stfts.shape)

        # Keep only the desired frequency bins
        stfts = stfts[:, valid_bins, :15]  # Ensure exactly 7 time frames
        print(stfts.shape)

        # Separate real and imaginary parts
        real_part = np.real(stfts)
        imag_part = np.imag(stfts)

        # Interleave real and imaginary parts into 8 channels
        output_data[i, 0::2] = real_part  # Even indices store real parts
        output_data[i, 1::2] = imag_part  # Odd indices store imaginary parts

    return output_data

print(audio_data.shape)
stfts = []
for i in range(len(audio_data)):
    audio_split = audio_data[i].reshape((1,audio_data.shape[1],audio_data.shape[2]))
    print("the shape of the audio_data is ", audio_split.shape)
    stft = process_audio(audio_split)
    stfts.append(stft)

stfts = np.array(stfts)
stfts = np.squeeze(stfts,axis = 1)
print(stfts.shape)

# save the stft data to h5 file 
with h5py.File(h5_file_path, 'a') as h5file:
    h5file.create_dataset('stft', data=stfts)

# stft_h5 = r"C:\Users\grizi\Desktop\TUD\year2\thesis\neural_network\DoA_Net\data\stft.h5"

# with h5py.File(stft_h5, 'w') as h5file1:
#     h5file1.create_dataset('stft', data=stfts)
#     h5file1.create_dataset('audio', data=audio_data)
#     h5file1.create_dataset('label', data=label_data)

# print("Processing complete. Data shape:", stfts.shape)
