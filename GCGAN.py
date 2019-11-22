from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

print(tf.__version__)

import glob  # 文件相关
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL  # 图像处理标准库

from tensorflow.keras import layers
# from tensorflow_core.python.keras import layers

import time
from IPython import display
CODE_ROOT_DIR = '/home/lnn/home/lnn/Code/GAN'


'''
加载、准备数据集
'''
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
# print(train_images.size)
# print(train_images.shape)
#
# # output
# # 47040000
# # (60000, 28, 28)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # 归一化至[-1,1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256



print(CODE_ROOT_DIR+'/dir-test')
# 批量化和打乱数据集
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

'''
生成器
'''
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                     activation='tanh'))

    assert model.output_shape == (None, 28, 28, 1)
    return model


'''
使用尚未训练的生成器生成一张图片
'''
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_img = generator(noise, training=False)
plt.imshow(generated_img[0, :, :, 0], cmap='gray')
plt.show()

'''
判别器
'''
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters=64,kernel_size=(5,5),padding='same',strides=(2,2),input_shape=[28,28,1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(filters=128,kernel_size=(5,5),padding='same',strides=(2,2)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# 使用尚未训练的判别器来对图片的真伪进行判断
discriminator = make_discriminator_model()
decision = discriminator(generated_img)
print(decision)


'''
定义损失函数和优化器
'''
# 当交叉熵越小说明二者之间越接近
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# 判别器损失
def discriminator_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output),real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output),fake_output)
    total_loss = real_loss+fake_loss
    return total_loss

# 生成器损失
def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output),fake_output)
    return loss

# 生成器和判别器的优化器
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

'''
保存检查点
'''
checkpoint_dir = CODE_ROOT_DIR+'/GCGAN_trianing_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt')
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer= discriminator_optimizer,
                                 generator = generator,
                                 discriminator = discriminator)

'''
定义训练循环
'''
EPOCHS = 50
noise_dim=100
num_example_to_generate = 16
seed = tf.random.normal([num_example_to_generate,noise_dim])

@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE,noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise,training=True)

        fake_output = discriminator(generated_images, training=True)
        real_output = discriminator(real_images,training=True)

        gen_loss= generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output,fake_output)

    gen_grad = gen_tape.gradient(gen_loss,generator.trainable_variables)
    disc_grad = disc_tape.gradient(disc_loss,discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gen_grad,generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_grad,discriminator.trainable_variables))

def train(dataset,epochs):
    for epoch in range(epochs):
        start_time = time.time()

        # 更新一次生成器和判别器
        for image_batch in dataset:
            train_step(image_batch)

        # 使用固定的测试集生成变化图
        display.clear_output(wait=True)
        generate_and_save(generator,epoch+1,seed)

        # 每15个epoch保存一次模型
        if (epoch+1)%15 ==0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch+1,time.time()-start_time))

    # 最后一个epoch结束后保存 图片
    display.clear_output(wait=True)
    generate_and_save(generator,epochs,seed)

# 生成并保存图片
def generate_and_save(model,epoch,test_input):
    predictions = model(test_input,training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0]*127.5+127.5,cmap='gray')
        plt.axis('off')
    plt.savefig(CODE_ROOT_DIR+'/pic/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


'''
训练模型
'''
train(train_dataset,EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

'''
创建GIF
'''
def display_image(epoch_no):
    return PIL.Image.open(CODE_ROOT_DIR+'/pic/image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('*/*/image*.png')
  print(filenames[0:5])
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import IPython
if IPython.version_info > (6,2,0,''):
  display.Image(filename=anim_file)