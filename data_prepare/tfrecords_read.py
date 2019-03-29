import tensorflow as tf
# image后面需要使用
from PIL import Image

# 手动输入路径
cwd = 'D:/PyCharm/PycharmProjects/AJ_Recognition/data_prepare/pic/inputdata/'


# 读入AJ_train.tfrecords
def read_and_decode(filename):
    # 生成一个queue队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       # 将image数据和label取出来
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    # reshape为224*224的3通道图片
    img = tf.reshape(img, [64, 64, 3])
    # 归一化
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # 在流中抛出label张量
    label = tf.cast(features['label'], tf.int32)
    return img, label


# 使用函数读入流中
image, label = read_and_decode("AJ_train.tfrecords")
# 开始一个会话
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(80):
        # 在会话中取出image和label
        example, l = sess.run([image, label])
        # 这里Image是之前提到的
        img = Image.fromarray(example, 'RGB')
        # 存下图片
        img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')
        print('----------------------------')
        print(example, l)
    coord.request_stop()
    coord.join(threads)
