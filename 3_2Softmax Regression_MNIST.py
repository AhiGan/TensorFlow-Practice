import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # one_hot=True 编码以概率分布的形式读“单类（只有一个label的值为1）”的标记情况

# 输出数据信息
print(mnist.train.images.shape, mnist.train.labels.shape)  # 训练集
print(mnist.validation.images.shape, mnist.validation.labels.shape)  # 验证集
print(mnist.test.images.shape, mnist.test.labels.shape)  # 测试集

'''
输出为
(55000, 784) (55000, 10)
(5000, 784) (5000, 10)
(10000, 784) (10000, 10)
784 = 28 * 28 等于将28*28的灰度图像拉成784的一维向量，
这里丢弃了图片的空间结构信息，是因为学习任务比较简单所以进行了简化
'''

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])  # 第二个参数是输入数据的shape，不限条数，每条是784维的向量
W = tf.Variable(tf.zeros([784, 10]))  # 权重
b = tf.Variable(tf.zeros([10]))  # bias

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 使用cross-entropy作为loss function
y_ = tf.placeholder(tf.float32, [None, 10])  # 真实标记
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))  # 横向加和 reduction_indices=[1]
# reduction_indices参数，表示函数的处理维度

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 学习率设为0.5，优化目标是最小化交叉熵
tf.global_variables_initializer().run()  # 全局变量初始化

for i in range(1000):  # epoch
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 构建mini-batch
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

# 0.9175
