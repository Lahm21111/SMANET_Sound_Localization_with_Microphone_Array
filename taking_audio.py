import rclpy
from rclpy.node import Node
from cae_microphone_array.msg import AudioStream
import h5py
import numpy as np

class AudioArraySubscriber(Node):
    def __init__(self):
        super().__init__('audio_array_subscriber')
        self.subscription = self.create_subscription(
            AudioStream,
            '/cae_micarray/audio/array',
            self.listener_callback,
            10
        )
        
        # 创建 HDF5 文件，初始化数据集
        self.h5_file = h5py.File('audio_data.h5', 'w')
        self.h5_dataset = self.h5_file.create_dataset("audio_data", shape=(0, 56, 16384), maxshape=(None, 56, 16384), dtype='float32')
        self.sequence = 0

        # 用于临时存储每 5 个消息的数据
        self.temp_data = []
        print('Subscriber initialized')

    def listener_callback(self, msg):
        # 将当前消息的数据转换为 numpy 数组并追加到临时列表
        
        data = np.array(msg.data).reshape(56,-1)
        
        # 获取当前数据集的大小，并调整数据集的形状
        current_size = self.h5_dataset.shape[0]
        new_size = current_size + 1  # 计算新大小
        self.h5_dataset.resize((new_size, 56, 16384))  # 扩展数据集的大小
        
        # 清空临时数据列表
        self.h5_dataset[current_size:new_size, :, :] = data
        self.temp_data = []
        self.get_logger().info('Batch of 5 audio messages appended to h5 file.')

    def close_h5_file(self):
        # 保存剩余的数据（如果存在）
        # if self.temp_data:
        #     batch_data = np.concatenate(self.temp_data)
        #     print(batch_data.size)

        #     new_size = current_size + batch_data.size
        #     self.h5_dataset[current_size:new_size] = batch_data
        #     self.get_logger().info('Remaining audio messages appended to h5 file.')

        # 关闭 HDF5 文件
        self.h5_file.close()

def main(args=None):
    rclpy.init(args=args)
    audio_array_subscriber = AudioArraySubscriber()
    
    try:
        rclpy.spin(audio_array_subscriber)
    except KeyboardInterrupt:
        audio_array_subscriber.get_logger().info('Shutting down...')
    finally:
        audio_array_subscriber.close_h5_file()
        audio_array_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
