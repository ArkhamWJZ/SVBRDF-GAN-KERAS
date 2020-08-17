import tensorflow as tf
import numpy as np
import collections

import render

Examples = collections.namedtuple("Examples", "iterator, concats")

def gaussian_kernel(size=5,sigma=2):
    """ 二维高斯曲面

    Args:
        size (int, optional): 模版形状. Defaults to 5.
        sigma (int, optional): 高斯曲面的sigma. Defaults to 2.

    Returns:
        np.narray : 二维高斯模版 size*size
    """
    x_points = np.arange(-(size-1)//2,(size-1)//2+1,1)
    y_points = x_points[::-1]
    xs,ys = np.meshgrid(x_points,y_points)
    kernel = np.exp(-(xs**2+ys**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
    return kernel/kernel.sum()

def scale(imgtile):
    # =>[0,1]
    minpixel = tf.reduce_min(imgtile,axis=[1,2],keepdims=True)
    maxpixel = tf.reduce_max(imgtile,axis=[1,2],keepdims=True)
    scaleimg = (imgtile - minpixel)/(maxpixel - minpixel+0.0001)
    return scaleimg

def blur(x,kernel):
    """在图片的三个通道上进行高斯模糊

    Args:
        x (tensor): 输入图片
        kernel (np.narray): 二维高斯滤波模版

    Returns:
        xrgb_blur (tensor): 经过高斯滤波的三通道图片
    """
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    xr,xg,xb = tf.expand_dims(x[:,:,:,0],-1), tf.expand_dims(x[:,:,:,1],-1), tf.expand_dims(x[:,:,:,2],-1)

    xr_blur = tf.nn.conv2d(xr, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xg_blur = tf.nn.conv2d(xg, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xb_blur = tf.nn.conv2d(xb, kernel, strides=[1, 1, 1, 1], padding='SAME')

    xrgb_blur = tf.concat([xr_blur, xg_blur, xb_blur], axis=3)
    return xrgb_blur

def normalize_aittala(input_img,kernel1):
    """ Get the ground true which will be used to calculate the diffuseloss(mini-variance)

    Args:
        input_img (tensor): 输入图片
        kernel1 (np.narray): 二维高斯滤波核（用于高斯模糊）

    Returns:
        scaley (tensor): X^* 用于计算漫反射误差（最小平方）
    """
    y_mean = blur(input_img,kernel1)
    y_stddv = tf.sqrt(blur(tf.square(input_img - y_mean), kernel1))
    norm = (input_img - y_mean)/(y_stddv+0.0001)
    scaley = scale(norm)
    return scaley

def concat_inputs(filename,kernel1):
    """ 获得输入图片，并进行预处理，shader vector分布，intensity分布
        漫反射ground true标记计算
    Args:
        filename (string): 输入图片路径名
        kernel1 (np.narray): 二维高斯模糊的滤波核

    Returns:
        img_tobe_sliced（tensor）: [flash_input,wv,inten,initdiffuse] 3*4通道
    """
    flashimg_string = tf.io.read_file(filename)
    flash_input = tf.image.decode_image(flashimg_string)
    flash_input = tf.image.convert_image_dtype(flash_input, dtype=tf.float32)
    flash_input = tf.expand_dims(flash_input**2.2,axis=0)

    initdiffuse = normalize_aittala(flash_input,kernel1)# guessed diffuse map

    img_h, img_w = flash_input.shape[1], flash_input.shape[2]
    wv,inten = render.generate_vl(img_w, img_h) # eview_vec, I

    img_tobe_sliced = tf.concat([flash_input,wv,inten,initdiffuse],axis=-1)

    return img_tobe_sliced

# 裁剪tile
def crop_imgs(raw_input, random_seed, tile_size):
    concat_tile = tf.image.random_crop(raw_input, size=[tile_size*2, tile_size*2, 12],seed=random_seed)
    return concat_tile
def load_examples(img_tobe_sliced, tilesize=256, BATCH_SIZE=1, seed=24):
    """裁剪待训练图片集为训练使用的切片，大小为 **tile_size*2

    Args:
        img_tobe_sliced (tensor): 待训练数据
        tilesize (int, optional): 想要输出的切片大小/2. Defaults to 256.

    Returns:
        (Examples): 返回数据集上切片的 iterator
    """
    dataset = tf.data.Dataset.from_tensor_slices(img_tobe_sliced)
    # 将输入随机裁剪为 [tile_size*2, tile_size*2,12] 大小的片段
    dataset = dataset.map(lambda x:crop_imgs(x, seed, tile_size = tilesize))
    dataset = dataset.repeat()
    batched_dataset = dataset.batch(BATCH_SIZE)

    iterator = iter(batched_dataset)
    concat_batch = iterator.get_next()
    return Examples(
        iterator = iterator,
        concats = concat_batch,
    )

def instancenorm(input):
    """ instance norm 对单张图片对数据处理 一个channel内做归一化，算H*W的均值，
        用在风格化迁移，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之
        间的独立。

    Args:
        input (tensor): 待训练图像

    Returns:
        normalized, mean, variance (tensor):
    """
    input = tf.identity(input)
    channels = input.get_shape()[3]
    offset = tf.Variable(tf.zeros([1,1,1, channels]), dtype=tf.float32,name="offset")
    scale = tf.Variable(tf.random_normal_initializer(1.0, 0.02)(shape=[1,1,1, channels], dtype=tf.float32), name="scale")

    mean, variance = tf.nn.moments(input, axes=[1, 2], keepdims=True)
    variance_epsilon = 1e-5
    normalized = (((input - mean) / tf.sqrt(variance + variance_epsilon)) * scale) + offset

    return normalized, mean, variance

def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keepdims=True))
    return tf.math.divide(tensor, Length)
# input: height map, [bs, h, w, 1], (0, 1)
# output: normal map, [bs, h, w, 3], normalized, (-1, 1)
def height_to_normal(height):

    dx = height[:, 1:, :, :] - height[:, :-1, :, :]
    dx_zeros_shape = (height.shape[0], 1, height.shape[2], height.shape[3])
    dx_zeros = tf.zeros(dx_zeros_shape)
    dx = tf.concat([dx, dx_zeros], axis=1)

    dy = height[:, :, 1:, :] - height[:, :, :-1, :]
    dy_zeros_shape = (height.shape[0], height.shape[1], 1, height.shape[3])
    dy_zeros = tf.zeros(dy_zeros_shape)
    dy = tf.concat([dy,dy_zeros],axis=2)

    # dx, dy = tf.image.image_gradients(height)
    c1 = 32; c2 = 32
    ddx = c1 * dy; ddy = c2 * dx

    one = tf.ones_like(ddx)
    n = tf.concat([-ddx, -ddy, one], axis=-1)
    n = tf_Normalize(n)
    return n