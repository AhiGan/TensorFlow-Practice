# check if the GPU is working properly

import tensorflow as tf

a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
print(a)

b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)                                  # 判断GPU是否可以用

print(b)

config=tf.ConfigProto(log_device_placement=True)
print(config)

# import tensorflow as tf
#
# a = tf.random_normal((100, 100))
# b = tf.random_normal((100, 500))
# c = tf.matmul(a, b)
# sess = tf.InteractiveSession()
# print(sess.run(c))
