import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 标准的均匀分布的Xavier初始化器
def xavier_init(fan_in, fan_out, constant=1):
    low = - constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,maxval=high,dtype=tf.float32)

# 去噪自编码器的class
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus
                 , optimizer = tf.train.AdamOptimizer, scale = 0.1):  # transfer_function-隐藏层激活函数，scale-高斯噪声系数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale= tf.placeholder(tf.float32)  # ？？
        self.train_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 输入
        self.x = tf.placeholder(tf.float32,[None,self.n_input])

        # 隐藏层输出
        # self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
        #         self.weights['w1']),
        #         self.weights['b1']))
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + self.train_scale * tf.random_normal((self.n_input,)),
                                                     self.weights['w1']),self.weights['b1']))  # scale

        # 重构
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),self.weights['b2'])

        # loss-平方误差
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x,self.reconstruction),2.0))

        # 优化器
        self.opimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess =tf.Session()
        self.sess.run(init)

    # 权重初始化，最后一层没有激活函数，因此不需要用xavier_init
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden]),dtype=tf.float32)
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input]),dtype=tf.float32)
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input]),dtype=tf.float32)
        return all_weights

    # 用一个batch数据进行训练，并返回当前的损失
    def partial_fit(self,X):
        cost, opt = self.sess.run((self.cost,self.opimizer),feed_dict={self.x:X,self.scale:self.train_scale})
        return cost

    # 在测试集上对模型性能进行评测时求cost不触发训练
    def calc_total_cost(self, X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.train_scale})

    # 返回隐藏层的输出结果
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X,self.scale:self.train_scale})

    # 复原函数,和transform将整个自编码器拆分为两部分
    def generate(self,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights['b1'])

        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})


    # 复原过程，包裹提取高阶特征、通过高阶特征复原为原始数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.train_scale})

    # 获取隐藏层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐藏层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


# 数据预处理：标准化
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 不放回批抽样
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

# 读入数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# 标准化
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 设置训练参数
n_samples = int(mnist.train.num_examples)
training_epoches = 20
batch_size = 128
display_step = 1

# 创建一个自编码器实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,n_hidden=200,
                                               transfer_function= tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),scale=0.01)

# 训练
for epoch in range(training_epoches):
    avg_cost = 0.   # 每一个epoch 的平均损失
    total_batch = int(n_samples / batch_size)
    for j in range(total_batch):
        batch_x = get_random_block_from_data(X_train,batch_size)
        cost = autoencoder.partial_fit(batch_x)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "Average Cost:", "{:.9f}".format(avg_cost))   # “%04d” 以四位小数的形式输出，不足四位小数时用0补齐


print("Total Cost:" + str(autoencoder.calc_total_cost(X_test)))

