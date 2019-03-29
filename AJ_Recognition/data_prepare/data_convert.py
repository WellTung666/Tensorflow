import os
import tensorflow as tf
from PIL import Image

cwd = 'D:/PyCharm/PycharmProjects/AJ_Recognition/data_prepare/pic/'

classes = {'AJ1', 'AJ4', 'AJ11', 'AJ12'}
# 要生成的文件
writer = tf.python_io.TFRecordWriter("AJ_train.tfrecords")

for index, name in enumerate(classes):
    class_path = cwd+name+'/'
    for img_name in os.listdir(class_path):
        # 每一个图片的地址
        img_path = class_path+img_name

        img = Image.open(img_path)
        # 设置图片需要转化成的大小
        img = img.resize((64, 64))
        # 将图片转化为二进制格式
        img_raw = img.tobytes()

        # example对象对label和image数据进行封装
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        # 序列化为字符串
        writer.write(example.SerializeToString())

writer.close()
