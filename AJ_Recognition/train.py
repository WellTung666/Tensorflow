import os
import numpy as np
import tensorflow as tf
import batch
import model


# 变量声明
N_CLASSES = 4
IMG_W = 64  # resize图像，太大的话训练时间久
IMG_H = 64
BATCH_SIZE = 20     # 每个batch要放多少张图片
CAPACITY = 200      # 一个队列最大多少
MAX_STEP = 1000  # 一般大于10K
learning_rate = 0.0001  # 一般小于0.0001

# 获取批次batch
train_dir = 'D:/PyCharm/PycharmProjects/AJ_Recognition/data_prepare/pic/inputdata'  # 训练样本的读入路径
logs_train_dir = 'D:/PyCharm/PycharmProjects/AJ_Recognition/train_log'   #logs存储路径

train, train_label, val, val_label = batch.get_files(train_dir, 0.3)
# 训练数据及标签
train_batch, train_label_batch = batch.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
# 测试数据及标签
val_batch, val_label_batch = batch.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)


# 训练操作定义
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = model.losses(train_logits, train_label_batch)
train_op = model.trainning(train_loss, learning_rate)
train_acc = model.evaluation(train_logits, train_label_batch)

# 测试操作定义
test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = model.losses(test_logits, val_label_batch)
test_acc = model.evaluation(test_logits, val_label_batch)



# 这个是log汇总记录
summary_op = tf.summary.merge_all()

# 产生一个会话
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
# 产生一个saver来存储训练好的模型
saver = tf.train.Saver()
# 所有节点初始化
sess.run(tf.global_variables_initializer())
# 队列监控
coord = tf.train.Coordinator()  # 设置多线程协调器
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 进行batch的训练
try:
    # 执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        # 启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

        # 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
        if step % 100 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str, step)
        checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
        saver.save(sess, checkpoint_path)

        # 每隔100步，保存一次训练好的模型
        if (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()
coord.join(threads)
sess.close()


