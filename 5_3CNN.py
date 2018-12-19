import os
os.environ["CUDA_VISIABLE_DEVICES"]="0"


import tensorflow as tf
import cifar10, cifar10_input

import numpy as np
import time

max_steps = 3000
batch_size = 128

# 准备数据
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

# 下载数据集并解压到指定位置
cifar10.maybe_download_and_extract()

# 获取数据增强后的训练数据
images_train, labels_train = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

# 获取测试数据
images_test, labels_test = cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

# weight的初始化
def variable_with_weight_loss(shape,stddev,w1):  # w1是L2正则项的权重
    var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 数据的placeholder
image_holder = tf.placeholder(tf.float32,[batch_size,24,24,3]) # [一个batch的样本个数，样本size1，样本size2，通道数]
label_holder = tf.placeholder(tf.int32,[batch_size])


# 网络结构
# 第一个卷积层
weight1 = variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
kernel1 = tf.nn.conv2d(image_holder,weight1,strides=[1,1,1,1],padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)  # 后面的参数分别是什么意思？


# 第二个卷积层；bias设为0.1，先LRN再最大池化
weight2 = variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
kernel2 = tf.nn.conv2d(norm1,weight2,strides=[1,1,1,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75)
pool2 = tf.nn.max_pool(norm2,[1,3,3,1],[1,2,2,1],padding='SAME')

# 第一个全连接层，含有384个隐藏节点
reshape = tf.reshape(pool2,shape=[batch_size,-1])  # 将每个样本都平铺为1维向量
dim = reshape.get_shape()[1].value  # 获取列的个数
weight3 = variable_with_weight_loss([dim, 384],stddev=0.04,w1=0.004)  # 隐藏结点为384个，为防止过拟合加L2正则项
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)

#第二个全连接层，含有192个隐藏节点
weight4 = variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4)

#模型Inference的输出结果，这里没有softmax，因为其被整合到后面的loss部分
weight5 = variable_with_weight_loss(shape=[192,10],stddev=1/192.0,w1=0.0)
bias5 =tf.Variable(tf.constant(0.0,shape=[10]))
logits = tf.add(tf.matmul(local4,weight5),bias5)
# inference部分到此结束

# loss部分
#定义计算总loss的函数
def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_losses')

#计算最终loss
loss = loss(logits,label_holder)

# 优化器
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# top k的准确率
top_k_op = tf.nn.in_top_k(logits,label_holder,1)  # 使用默认的top1

# 创建默认session及初始化
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 图片增强线程
tf.train.start_queue_runners()

# 训练部分
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    print(image_batch)
    print(image_batch.shape)
    _,loss_value = sess.run([train_op,loss],feed_dict={image_holder:image_batch,label_holder:label_batch})
    duration = time.time()-start_time

    # 每10个step显示一次loss和训练速度
    if step%10 == 0:
        examples_per_sec = batch_size/duration
        sec_per_batch = float(duration)
        format_str =('step %d, loss %.2f (%.1f examples/sec, %.3f sec/batch)')
        print(format_str %(step,loss_value, examples_per_sec, sec_per_batch))


# 评测模型
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples/batch_size))
true_count = 0
total_sample_count = num_iter*batch_size  # 如果不是刚好整除会有重复的示例被测试到？
step=0
while step<num_iter:
    image_batch, label_batch = sess.run([images_test,labels_test])
    print(image_batch)
    print(image_batch.shape)
    predictions =sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count/total_sample_count
print('precision @ 1 = %.3f' % precision)



