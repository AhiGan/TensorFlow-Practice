import os
os.environ["CUDA_VISIABLE_DEVICES"]="0"

import tensorflow as tf
import time
import math
from datetime import datetime

batch_size = 32
nums_batch = 100


# 展示每一个网络层输出的尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


'''
网络的推断部分，接受images作为输出，返回最后一层池化层和网络所有需要训练的参数
Conv+Relu、LRN、Pool、Conv+Relu、LRN、Pool、Conv+Relu、Conv+Relu、Conv+Relu、Pool、FC+Relu、FC+Relu、FC
'''


def inference(images):
    parameters = []

    # conv1：Conv+Relu
    with tf.name_scope('conv1')as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv1)

    # lrn1,后面参数就是使用的论文参数
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')

    # pool1
    pool1 = tf.nn.max_pool(lrn1, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    # conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

    # lrn2
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')

    # pool2
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 192, 384], stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool2, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]),trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv3)

    # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 384, 256], stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv3, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv4)

    # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 256, 256], stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv4, kernel, strides=[1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv5)

    # pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    # 正式使用时，需添加三个全连接层

    return pool5, parameters


'''
每轮计算时间的函数
'''


def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(nums_batch + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration_time = time.time() - start_time
        if i > num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration_time))
            total_duration += duration_time
            total_duration_squared += duration_time * duration_time

    mn = total_duration / nums_batch
    vr = total_duration_squared / nums_batch - mn * mn
    std = math.sqrt(vr)

    print('%s： %s across %d step, %.3f +/- %.3f sec / batch ' %
          (datetime.now(), info_string, nums_batch, mn, std))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.truncated_normal([batch_size,
                                                  image_size,
                                                  image_size,
                                                  3], dtype=tf.float32, stddev=1e-1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        time_tensorflow_run(sess, pool5, 'Forward')

        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, 'Fprward-bacward')


run_benchmark()
