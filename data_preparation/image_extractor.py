#!/usr/bin/env python3

import os
import rospy
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from extractor_node.msg import AvReader
import h5py
import numpy as np

def save_images_to_h5(bag_file_path, h5_file_path, topic_name):
    # 创建 cv_bridge 对象来将 ROS 图像消息转换为 OpenCV 格式
    bridge = CvBridge()
    image_data = []

    # 打开 rosbag 文件
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            # 检查消息是否为图像消息
            if isinstance(msg, Image):
                try:
                    # 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 图像
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    image_data.append(cv_image)
                except Exception as e:
                    rospy.logwarn(f"Failed to process image at {t}: {e}")

    # 将所有图像写入 .h5 文件
    with h5py.File(h5_file_path, 'w') as h5_file:
        h5_file.create_dataset("images", data=np.array(image_data), compression="gzip")
        rospy.loginfo(f"Saved {len(image_data)} images to {h5_file_path}")

if __name__ == "__main__":
    rospy.init_node("image_saver_to_h5_node", anonymous=True)

    # 参数：rosbag 文件路径，输出 .h5 文件路径，图像话题名称
    bag_file_path = rospy.get_param("~bag_file_path", "path/to/your/rosbag.bag")
    h5_file_path = rospy.get_param("~h5_file_path", "path/to/output_file.h5")
    topic_name = rospy.get_param("~topic_name", "/camera/image_raw")

    save_images_to_h5(bag_file_path, h5_file_path, topic_name)
