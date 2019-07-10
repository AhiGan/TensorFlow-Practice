import os
import tensorflow as tf
from tensorflow.python.estimator.inputs import numpy_io
import numpy as np
import collections
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow import feature_column

from tensorflow.python.feature_column.feature_column import _LazyBuilder

'''
将feature转为数值列
tf.feature_column.numeric_column(
    key,
    shape=(1,),
    default_value=None,
    dtype=tf.float32,
    normalizer_fn=None
)
'''


def test_numeric():
    price = {'price': [[1.], [2.], [3.], [4.]]}
    builder = _LazyBuilder(price)

    def transform_fn(x):
        return x + 2

    price_column = feature_column.numeric_column('price', normalizer_fn=transform_fn)
    price_transformed_tensor = price_column._get_dense_tensor(builder)

    with tf.Session() as session:
        print(session.run([price_transformed_tensor]))

    # 使用feature_column.input_layer，price_column._get_dense_tensor两者结果一致
    price_transformed_tensor_input_layer = feature_column.input_layer(price, [price_column])
    with tf.Session() as session:
        print('use input_layer' + '_' * 40)
        print(session.run([price_transformed_tensor_input_layer]))


# test_numeric()


'''
对数值数据，根据桶边界进行分桶编码
Bucketized column用来把numeric column的值按照提供的边界（boundaries)离散化为多个值。离散化是特征工程常用的一种方法
tf.feature_column.bucketized_column(
    source_column,
    boundaries
)
'''


def test_bucketized_column():
    sample = {'price': [[5.], [16], [25], [36]], 'time': [[2.], [6], [8], [15]]}
    price_column = feature_column.numeric_column('price')
    bucket_price = feature_column.bucketized_column(price_column, [10, 20, 30, 40])
    price_bucket_tensor = feature_column.input_layer(sample, [bucket_price])

    time_column = feature_column.numeric_column('time')
    bucket_time = feature_column.bucketized_column(time_column, [5, 10, 12])
    time_bucket_tensor = feature_column.input_layer(sample, [bucket_time])
    with tf.Session() as session:
        print(session.run([price_bucket_tensor, time_bucket_tensor]))


# test_bucketized_column()


'''
# 分类识别列：将数值数据转为独热编码,注意特征里的数值数据需小于num_buckets
# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)
    
'''


def test_identity_feature_column():
    sample = {'price': [[1], [2], [3], [0]]}
    # price_column = feature_column.numeric_column('price')
    price_column = feature_column.categorical_column_with_identity(key='price', num_buckets=4)
    indicator = feature_column.indicator_column(price_column)
    price_column_tensor = feature_column.input_layer(sample, [indicator])

    with tf.Session() as session:
        print(session.run([price_column_tensor]))


# test_identity_feature_column()


'''
# 分类词汇列：将文本数据根据单词列表生成为one-shot特征列
categorical_column_with_vocabulary_list(
    key,
    vocabulary_list,
    dtype=None,
    default_value=-1,
    num_oov_buckets=0
)
num_oov_buckets 大于0时，对不存在于vocabulary list中的词汇在[vocabulary_list_size,vocabulary_list_size+num_oov_buckets]中产生
如果种类大于num_oov_buckets，会有重复，但每一种的label会保持一致
在tensorflow中有两种提供词汇表的方法，一种是用list，另一种是用file，对应的feature column分别为：

tf.feature_column.categorical_column_with_vocabulary_list
tf.feature_column.categorical_column_with_vocabulary_file
'''


def test_categorical_column_with_vocabulary_list():
    pets = {'pets': ['rabbit', 'pig', 'dog', 'cat', 'mouse', 'over1', 'over2', 'over1', 'over3', 'over2', 'mouse']}
    column = tf.feature_column.categorical_column_with_vocabulary_list(
        key='pets', vocabulary_list=['cat', 'dog', 'rabbit', 'pig'],
        dtype=tf.string, default_value=-1, num_oov_buckets=3)
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))


# test_categorical_column_with_vocabulary_list()


def test_categorical_column_with_vocabulary_file():
    pets = {'pets': ['rabbit', 'pig', 'dog', 'cat', 'mouse', 'over1', 'over2', 'over1', 'over3', 'over2', 'mouse']}

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fc_path = os.path.join(dir_path, 'pets_fc.txt')

    column = tf.feature_column.categorical_column_with_vocabulary_file(
        key='pets', vocabulary_file=fc_path,
        num_oov_buckets=3)
    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(pets, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))


# test_categorical_column_with_vocabulary_file()


'''
Hash栏 Hashed Column
categorical_column_with_hash_bucket(
    key,
    hash_bucket_size,
    dtype=tf.string
)
箱子数量的选择很重要，越大获得的分类结果越精确
'''


def test_categorical_column_with_hash_bucket():
    colors = {'colors': ['green', 'red', 'blue', 'yellow', 'pink', 'blue', 'red', 'indigo']}

    column = tf.feature_column.categorical_column_with_hash_bucket(
        key='colors',
        hash_bucket_size=5,
    )

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(colors, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))


# test_categorical_column_with_hash_bucket()

'''
tf.feature_column.crossed_column(
    keys,
    hash_bucket_size,
    hash_key=None
)
交叉列可以把多个特征合并成为一个特征，比如把经度longitude、维度latitude两个特征合并为地理位置特征location
'''


def test_crossed_column():
    featrues = {
        'longtitude': [1, 35, 68, 50, 9, 45,20],
        'latitude': [1, 35, 68, 51, 81, 24,20]
    }

    longtitude = tf.feature_column.numeric_column('longtitude')
    latitude = tf.feature_column.numeric_column('latitude')

    longtitude_b_c = tf.feature_column.bucketized_column(longtitude, [33, 66])
    latitude_b_c = tf.feature_column.bucketized_column(latitude, [33, 66])

    column = tf.feature_column.crossed_column([longtitude_b_c, latitude_b_c], 12)

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(featrues, [indicator])
    tensor1 = tf.feature_column.input_layer(featrues, [longtitude_b_c])

    tensor2 = tf.feature_column.input_layer(featrues, [latitude_b_c])


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))
        print(session.run([tensor1]))
        print(session.run([tensor2]))


# test_crossed_column()


'''
指示列Indicator Columns和嵌入列Embeding Columns
指示列并不直接操作数据，但它可以把各种分类特征列转化成为input_layer()方法接受的特征列。
当我们遇到成千上万个类别的时候，独热列表就会变的特别长[0,1,0,0,0,....0,0,0]。嵌入列可以解决这个问题，它不再限定每个元素必须是0或1，而可以是任何数字，从而使用更少的元素数表现数据。
embedding_column(
    categorical_column,
    dimension,
    combiner='mean',
    initializer=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)
嵌入列表的维数等于类别总数开4次方,如81类别的嵌入列为3元
categorical_column: 使用categoryical_column产生的sparsor column
dimension: 定义embedding的维数
combiner: 对于多个entries进行的推导。默认是meam, 但是 sqrtn在词袋模型中，有更好的准确度。
initializer: 初始化方法，默认使用高斯分布来初始化。
tensor_name_in_ckpt: 可以从check point中恢复
ckpt_to_load_from: check point file，这是在 tensor_name_in_ckpt 不为空的情况下设置的.
max_norm: 默认是l2
trainable: 是否可训练的，默认是true
可以用-1来填充一个不定长的ID序列，这样可以得到定长的序列，然后经过embedding column之后，填充的-1值不影响原来的结果
'''

def test_embedding():
    tf.set_random_seed(1)
    color_data = {'color': [['R', 'G'], ['G', 'A'], ['B', 'B'], ['A', 'A']]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B'], dtype=tf.string, default_value=-1
    )

    color_column_tensor = color_column._get_sparse_tensors(builder)


    color_embeding = feature_column.embedding_column(color_column, 4, combiner='sum')
    color_embeding_dense_tensor = feature_column.input_layer(color_data, [color_embeding])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([color_column_tensor.id_tensor]))
        print('embeding' + '_' * 40)
        print(session.run([color_embeding_dense_tensor]))

# test_embedding()


'''
多个特征共享相同的embeding映射空间
tf.feature_column.shared_embedding_columns(
    categorical_columns,
    dimension,
    combiner='mean',
    initializer=None,
    shared_embedding_collection_name=None,
    ckpt_to_load_from=None,
    tensor_name_in_ckpt=None,
    max_norm=None,
    trainable=True
)
categorical_columns 为需要共享embeding映射空间的类别特征列表
其他参数与embedding column类似
'''
def test_shared_embedding_column_with_hash_bucket():
    color_data = {'color': [[2, 2], [5, 5], [0, -1], [0, 0]],
                  'color2': [[2], [5], [-1], [0]]}  # 4行样本
    builder = _LazyBuilder(color_data)
    color_column = feature_column.categorical_column_with_hash_bucket('color', 7, dtype=tf.int32)
    color_column_tensor = color_column._get_sparse_tensors(builder)
    color_column2 = feature_column.categorical_column_with_hash_bucket('color2', 7, dtype=tf.int32)
    color_column_tensor2 = color_column2._get_sparse_tensors(builder)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('not use input_layer' + '_' * 40)
        print(session.run([color_column_tensor.id_tensor]))
        print(session.run([color_column_tensor2.id_tensor]))

    # 将稀疏的转换成dense，也就是one-hot形式，只是multi-hot
    color_column_embed = feature_column.shared_embedding_columns([color_column2, color_column], 3, combiner='sum')
    print(type(color_column_embed))
    color_dense_tensor = feature_column.input_layer(color_data, color_column_embed)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print('use input_layer' + '_' * 40)
        print(session.run(color_dense_tensor))

# test_shared_embedding_column_with_hash_bucket()
# 输出两个3列的embedding后的矩阵，可以观察到color2的-1样本对应的embedding为全零，其他列color的embedding为color2的embedding的两倍


'''
Weighted categorical column
给一个类别特征赋予一定的权重，比如给用户行为序列按照行为发生的时间到某个特定时间的差来计算不同的权重
tf.feature_column.weighted_categorical_column(
    categorical_column,
    weight_feature_key,
    dtype=tf.float32
)
'''

def test_weighted_categorical_column():
    features = {'color': [['R'], ['A'], ['G'], ['B'], ['R']],
                'weight': [[1.0], [5.0], [4.0], [8.0], [3.0]]}

    color_f_c = tf.feature_column.categorical_column_with_vocabulary_list(
        'color', ['R', 'G', 'B', 'A'], dtype=tf.string, default_value=-1
    )

    column = tf.feature_column.weighted_categorical_column(color_f_c, 'weight')

    indicator = tf.feature_column.indicator_column(column)
    tensor = tf.feature_column.input_layer(features, [indicator])

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print(session.run([tensor]))

# test_weighted_categorical_column()


'''
线性模型LinearModel
对所有特征进行线性加权操作（数值和权重值相乘）
linear_model(
    features,
    feature_columns,
    units=1,
    sparse_combiner='sum',
    weight_collections=None,
    trainable=True
)
'''

def test_LinearModel():
    def get_linear_model_bias():
        with tf.variable_scope('linear_model', reuse=True):
            return tf.get_variable('bias_weights')

    def get_linear_model_column_var(column):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 'linear_model/' + column.name)[0]

    featrues = {
        'price': [[1.0], [5.0], [10.0]],
        'color': [['R'], ['G'], ['B']]
    }

    price_column = tf.feature_column.numeric_column('price')
    color_column = tf.feature_column.categorical_column_with_vocabulary_list('color',
                                                                             ['R', 'G', 'B'])
    prediction = tf.feature_column.linear_model(featrues, [price_column, color_column])

    bias = get_linear_model_bias()
    price_var = get_linear_model_column_var(price_column)
    color_var = get_linear_model_column_var(color_column)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        sess.run(bias.assign([7.0]))
        sess.run(price_var.assign([[10.0]]))
        sess.run(color_var.assign([[2.0], [2.0], [2.0]]))

        predication_result = sess.run([prediction])

        print(prediction)
        print(predication_result)

test_LinearModel()