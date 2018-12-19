import os

os.environ["CUDA_VISIABLE_DEVICES"] = "0"

from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
nums_batch = 100

'''
定义卷积函数，创建卷积层并把本层参数存入参数列表
'''


def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[n_out]), trainable=True, name='bias')
        z = tf.nn.bias_add(conv, bias)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, bias]
        return activation


'''
定义全连接层创建函数，这里的bias在书上没有写trainable=True
'''


def fc_op(input_op, n_out, name, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), trainable=True, name='bias')
        activation = tf.nn.relu_layer(input_op, kernel, bias, name=scope)
        p += [kernel, bias]
        return activation


'''
定义最大池化层的创建函数
'''


def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)


'''
创建VGGNet16的网络结构
'''


# inference部分
def inference_op(input_op, keep_prob):  # keep_prob为dropout的比率
    parameters = []

    # 第一段卷积
    conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=parameters)
    conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=parameters)
    pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dh=2, dw=2)

    # 第二段卷积
    conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=parameters)
    conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=parameters)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dh=2, dw=2)

    # 第三段卷积
    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=parameters)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=parameters)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=parameters)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dh=2, dw=2)

    # 第四段卷积
    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=parameters)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=parameters)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=parameters)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dh=2, dw=2)

    # 第五段卷积
    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=parameters)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=parameters)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=parameters)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dh=2, dw=2)

    # 平铺
    shape = pool5.get_shape()
    flattened_shape = shape[1].value * shape[2].value * shape[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

    # 第一层全连接
    fc6 = fc_op(resh1, n_out=4096, name='fc6', p=parameters)
    fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='fc6_dropout')

    # 第二层全连接
    fc7 = fc_op(fc6_drop, n_out=4096, name='fc7', p=parameters)
    fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='fc7_dropout')

    # 第三层全连接，输出层
    fc8 = fc_op(fc7_drop, n_out=1000, name='fc8', p=parameters)
    softmax = tf.nn.softmax(fc8)
    prediction = tf.argmax(softmax, 1)
    return prediction, softmax, fc8, parameters


'''
定义时间估计函数
'''


def time_tensorflow_run(session, target, feed, info_string):
    num_batch_burn_in = 10
    total_duration = 0.0
    total_duration_squard = 0.0

    for i in range(num_batch_burn_in + nums_batch):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i > num_batch_burn_in:
            if not i % 10:  # 每10轮输出一次当前的duration
                print('%s : step %d, duration = %.3f' %
                      (datetime.now(), i - num_batch_burn_in, duration))
            total_duration += duration
            total_duration_squard += duration * duration
    mn = total_duration / nums_batch
    vr = total_duration_squard / nums_batch - mn * mn
    std = math.sqrt(vr)
    print('%s: %s across %d, %.3f +/- %.3f sec/batch' %
          (datetime.now(), info_string, nums_batch, mn, std))


'''
测试VGGNet-16在TensorFlow上forward和backward的耗时
'''


def run_benchmark():
    with tf.Graph().as_default():
        # 构建输入数据
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        # 构建网络结构
        keep_prob = tf.placeholder(tf.float32)
        prediction, softmax, fc8, parameters = inference_op(images, keep_prob)
        # 构建Session，并进行全局初始化
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, prediction, {keep_prob: 1.0}, 'Forward')
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, 'Forward-backward')



# 主函数
run_benchmark()
print('test over')
