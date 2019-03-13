# 导入Tensorflow，加载MNIST数据
import tensorflow as tf
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data/", False, one_hot=True)

# 创建x，x是占位符，代表待识别的图片。
x = tf.placeholder("float", [None, 784])

# 在这里，我们都用全为零的张量来初始化W和b。因为我们要学习W和b的值，它们的初值可以随意设置。
# W是Softmax模型的参数，将一个784维的输入转化为一个10维的输出
W = tf.Variable(tf.zeros([784, 10]))
# b是一个Softmax模型的参数，一般叫“偏置值”，形状是[10]，所以我们可以直接把它加到输出上面。=
b = tf.Variable(tf.zeros([10]))

# y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)

# y_是实际的图像标签，同样以占位符表示，是以独热表示的。
y_ = tf.placeholder("float", [None, 10])

# 根据y和y_构造交叉熵损失
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 梯度下降，学习率0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 添加一个操作来初始化创建的变量
init = tf.initialize_all_variables()

# 创建一个Session。只有在Session中才能运行优化步骤train_step
sess = tf.Session()
sess.run(init)

# 开始训练模型，这里我们让模型循环训练1000次
# 该循环的每个步骤中，我们都会随机抓取训练数据中的100个批处理数据点，然后我们用这些数据点作为参数替换之前的占位符来运行train_step。
# batch_xs的形状为（100，784）的图像数据，batch_ys是形如（100，10）的实际标签。 batch_xs和batch_ys对应这占位符x和y_。
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
# 在session中运行train_step，运行是要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 下述y的值是在上述训练最后一步已经计算获得，所以能够与原始标签y_进行比较
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# 计算预测准确值，它们都是Tensor
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 这里是获取最终模型的准确率,准确率约为91%
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
