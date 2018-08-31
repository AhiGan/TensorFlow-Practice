import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

# 隐藏层参数初始化
in_units = 784
h1_units = 300
w1 = tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))  # tf.truncated_normal() 生成正太分布, 均值和方差自己设定
b1 = tf.Variable(tf.zeros([h1_units]))
w2 = tf.Variable(tf.zeros([h1_units,10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32,[None,in_units])
keep_prob = tf.placeholder(tf.float32)  # drop的比率在训练时小于1，测试时等于1

# 模型结构
hidden1 = tf.nn.relu(tf.matmul(x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob=keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop,w2)+b2)

# 定义损失函数和优化器
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# 训练步骤
tf.global_variables_initializer().run()
for epoch in range(3000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x:batch_x,y_:batch_y,keep_prob:0.75})

# 准确率评测
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))

# 0.979