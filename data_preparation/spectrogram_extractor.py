import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from matplotlib.backends.backend_pdf import PdfPages

def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        audio_data = h5_file['audio_data'][:]  # get the dataset from the file
        print("Data shape:", audio_data.shape)
        
        example_data = audio_data[0:5].reshape(56,-1)
        print(example_data.shape)
        
        # calculate_spectrogram
        sample_rate = 44100
        #save_spectrograms_to_pdf(example_data, sample_rate, "spectrogram.pdf")


def save_spectrograms_to_pdf(audio_data, sample_rate, filename="spectrograms.pdf"):
    # 创建 PDF 文件
    with PdfPages(filename) as pdf:
        for i in range(audio_data.shape[0]):  # 针对每个通道
            # 计算频谱图
            f, t, Sxx = spectrogram(audio_data[i], sample_rate)
            
            # 创建频谱图
            plt.figure(figsize=(10, 4))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Intensity [dB]')
            plt.title(f'Spectrogram - Channel {i + 1}')
            
            # 保存当前图像到 PDF 中
            pdf.savefig()
            plt.close()
    
    print(f"Spectrograms for all channels saved to {filename}")


# 调用读取函数并输出 shape
read_h5_file('/cae-microphone-array-containerized/src/Sound_Localization_with_Microphone_Array/data_preparation/audio_data.h5')

