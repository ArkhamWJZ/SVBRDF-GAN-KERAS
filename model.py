import tensorflow as tf
import numpy as np
import os
import net
import collections
import argparse

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,UpSampling2D
from tensorflow.keras.optimizers import Adam

import tool
import render


parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", required=True)
parser.add_argument("--log_dir", required=True)
parser.add_argument("--output_dir", required=True)

parser.add_argument("--crop_size", type = int, default=800)
parser.add_argument("--max_step", type = int, default=100)
parser.add_argument("--seed", type = int, default=24)

args,unknown = parser.parse_known_args()

tf.random.set_seed(args.seed)
np.random.seed(args.seed)

# Encoder En
# input :  encoder_inputs (1, 128, 128, 3)
# output : layers[-1] (1, 32, 32, 256)
class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.c_1 = Conv2D(9,5,1,padding="SAME",  kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_2 = Conv2D(64,3,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_3 = Conv2D(128,3,2,padding="SAME",kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_4 = Conv2D(256,3,2,padding="SAME",kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def call(self, inputs, training=False):
        x = self.c_1(inputs)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_2(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_3(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_4(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        return x

# Two Decoders
# De_{n,α} : produces normal and roughness
# input :    resout (1, 32, 32, 256)
#            outc int
# output :   layers[-1] (1, 256, 256, 2)
class Decoder_NR(Model):
    def __init__(self):
        super(Decoder_NR, self).__init__()
        self.c_1 = tf.keras.layers.Conv2D(512,3,1,padding="SAME")
        self.u_2 =   UpSampling2D(size=(2,2),interpolation='nearest')
        self.c_2_0 = Conv2D(256,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_2_1 = Conv2D(256,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.u_3 =   UpSampling2D(size=(2,2),interpolation='nearest')
        self.c_3_0 = Conv2D(64,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_3_1 = Conv2D(64,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.u_4 =   UpSampling2D(size=(2,2),interpolation='nearest')
        self.c_4_0 = Conv2D(2,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_4_1 = Conv2D(2,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def call(self, inputs, training=False):
        x = self.c_1(inputs)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.u_2(x)
        x = self.c_2_0(x)
        x = self.c_2_1(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.u_3(x)
        x = self.c_3_0(x)
        x = self.c_3_1(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.u_4(x)
        x = self.c_4_0(x)
        x = self.c_4_1(x)
        x = tf.nn.tanh(x)
        return x

# De_{ρd,ρs} : produces diffuse and specular.
# input :      resout (1, 32, 32, 256)
#              outc int
# output :     layers[-1] (1, 256, 256, 4)
class Decoder_DS(Model):
    def __init__(self):
        super(Decoder_DS, self).__init__()
        self.c_1 =   Conv2D(512,3,1,padding="SAME")
        self.u_2 =   UpSampling2D(size=(2,2),interpolation='nearest')
        self.c_2_0 = Conv2D(256,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_2_1 = Conv2D(256,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.u_3 =   UpSampling2D(size=(2,2),interpolation='nearest')
        self.c_3_0 = Conv2D(64,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_3_1 = Conv2D(64,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.u_4 =   UpSampling2D(size=(2,2),interpolation='nearest')
        self.c_4_0 = Conv2D(4,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_4_1 = Conv2D(4,4,1,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def call(self, inputs, training=False):
        x = self.c_1(inputs)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.u_2(x)
        x = self.c_2_0(x)
        x = self.c_2_1(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.u_3(x)
        x = self.c_3_0(x)
        x = self.c_3_1(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.u_4(x)
        x = self.c_4_0(x)
        x = self.c_4_1(x)
        x = tf.nn.tanh(x)
        return x
# input : generator_inputs (None, 256, 256, 3)
# output : output (None, 16, 16, 1)
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.c_1 = Conv2D(64,4,2,padding="SAME",  kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_2 = Conv2D(128,4,2,padding="SAME", kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_3 = Conv2D(256,4,2,padding="SAME",kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_4 = Conv2D(512,4,2,padding="SAME",kernel_initializer=tf.random_normal_initializer(0, 0.02))
        self.c_5 = Conv2D(1  ,4,1,padding="SAME",kernel_initializer=tf.random_normal_initializer(0, 0.02))

    def call(self, inputs, training=False):
        x = self.c_1(inputs)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_2(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_3(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_4(x)
        x,_,_ = tool.instancenorm(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        x = self.c_5(x)
        x = tf.nn.sigmoid(x)
        return x


class SVBRDF_GAN():
    def __init__(self):
        self.ganscale = 0.1
        self.inputsize = 128
        self.BATCH_SIZE = 1
        self.Examples = collections.namedtuple("Examples", "iterator, concats")
        self.lr = 0.00002

        # TODO: 跑的时候 size 更改成 101
        kernel = tool.gaussian_kernel(size=10, sigma=101)
        input_2b_sliced = tool.concat_inputs(args.input_dir,kernel)
        examples = tool.load_examples(input_2b_sliced, self.inputsize) # 裁剪 [flash_input,wv,inten,initdiffuse] 2*inputsize

        examples_flashes = examples.concats[:,:,:,0:3] # (1, 256, 256, 3)
        examples_inputs = tf.map_fn(lambda x:tf.image.central_crop(x, 0.5),elems=examples_flashes) # 留中心的50%

        wv = examples.concats[:,:,:,3:6]   # view light

        encoder_model = Encoder()
        discr_model = Discriminator()

        latentcode = encoder_model(examples_inputs) # Encoder En
        predictions, decoder_nr, decoder_ds = self.generator(latentcode) # 四个拼图 [norm, diffuse, roughness, specular]
        net_rerender = render.CTRender(predictions,wv,wv)

        dis_real = discr_model(examples_flashes)  # 真图片过判别器
        dis_fake = discr_model(net_rerender)      # 生成图过判别器

        dis_cost = self.patchGAN_d_loss(dis_fake,dis_real)  # 判别器loss function    d_loss = d_loss_real-1 + d_loss_fake-0
        gen_fake = self.patchGAN_g_loss(dis_fake)           # 生成器loss function part 1     d_loss_fake-1

        prediffuse = predictions[:,:,:,3:6]
        initd = examples.concats[:,:,:,9:12]
        diffuseloss = tf.reduce_mean(tf.abs(prediffuse - initd)) # 生成器loss function part 2    两个生成器一样的loss function
        gnr_cost = self.ganscale*(gen_fake) + diffuseloss
        gds_cost = self.ganscale*(gen_fake) + diffuseloss

        # TODO: 参数问题，源码中第一个是0
        optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999)

        discr_model.compile(
            optimizer = optimizer,
            loss = dis_cost,
            metrics=['accuracy']
        )
        decoder_nr.compile(
            optimizer = optimizer,
            loss = gnr_cost,
            metrics=['accuracy']
        )
        decoder_ds.compile(
            optimizer = optimizer,
            loss = gds_cost,
            metrics=['accuracy']
        )

        discr_model.fit_generator(examples.iterator, steps_per_epoch=1)
        # 是不是没有必要不停的保存中间图片以及中奖生成的模型、
        # 我们需要的只是，最后预测的四张生成图、渲染图、输入图
        # 中间生成器和判别器的准确率都可以打印出来

        # for epoch in range(args.max_step):

        #     for i in range(5):
        #         decoder_ds.train_on_batch()
        #         print("wanjianzhou")
        #         # 训练 gnr_optimizer
        #     for i in range(1):
        #         print("wanjianzhou")
        #         # 训练 gds_optimizer

        #     # 训练 d_optimizer

        #     if epoch % 500 == 0 or (epoch + 1) == args.max_step:
        #         # 打印损失
        #         print("wanjianzhou")

        #     if epoch % 1000 == 0 or (epoch + 1) == args.max_step:
        #         print("wanjianzhou")
        #         # 保存模型 ckpt 、保存图片

    def generator(self,latentz):
        """ 使用两个解码器进行SVBRDF分解，得到四个分量图

        Args:
            latentz (tensor): 经过Encoder编码之后的输入

        Returns:
            reconstructedOutputs (tensor): 分解后的四个分量的拼图，并做标准化 [-1, 1] => [0, 1]
        """
        decoder_nr = Decoder_NR()
        decoder_ds = Decoder_DS()
        OutputedHR = decoder_nr(latentz) # [1,256,256,2] 1+1
        OutputedDS = decoder_ds(latentz) # [1,256,256,4] 3+1

        partialOutputedheight = OutputedHR[:,:,:,0:1]
        normNormals = tool.height_to_normal((partialOutputedheight+1)/2)
        outputedRoughness = OutputedHR[:,:,:,1]
        outputedDiffuse = OutputedDS[:,:,:,0:3]
        outputedSpecular = OutputedDS[:,:,:,3]

        outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
        outputedRoughnessMap = tf.concat([outputedRoughnessExpanded,outputedRoughnessExpanded,outputedRoughnessExpanded],axis=-1)
        outputedSpecularExpanded = tf.expand_dims(outputedSpecular, axis = -1)
        outputedSpecularMap = tf.concat([outputedSpecularExpanded,outputedSpecularExpanded,outputedSpecularExpanded],axis=-1)

        reconstructedOutputs =  tf.concat([normNormals, outputedDiffuse, outputedRoughnessMap, outputedSpecularMap], axis=-1)

        return (reconstructedOutputs + 1)/2, decoder_nr, decoder_ds

    def patchGAN_d_loss(self, disc_fake, disc_real):
        loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
        loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
        disc_cost = loss_d_real + loss_d_fake
        return disc_cost

    def patchGAN_g_loss(self, disc_fake):
        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
        return gen_cost





if __name__ == "__main__":
    gan = SVBRDF_GAN()
