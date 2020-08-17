import tensorflow as tf
import numpy as np


def CTRender(M, eview_vec, elight_vec):
    """ 对输入的SVBRDF分图，使用微表面模型进行渲染

    Args:
        M (tensor): 输入SVBRDF四个图片
        eview_vec (tensor): 像素到摄像机的分布向量
        elight_vec (tensor):像素到光源的分布向量

    Returns:
        ctrender_batch (tensor): 渲染出的成品图
    """
    half_vec = (eview_vec + elight_vec) / 2
    half_vec_norm = tf.sqrt(tf.reduce_sum(tf.square(half_vec),axis=-1))
    half_vec_expand = tf.expand_dims(half_vec_norm,axis=-1)
    newhalf_vec = tf.concat([half_vec_expand,half_vec_expand,half_vec_expand],axis=-1)
    ehalf_vec = half_vec/newhalf_vec

    norm, diff, rough, spec = tf.split(M, 4, axis=3)

    norm = norm*2-1
    norm_mo = tf.sqrt(tf.reduce_sum(tf.square(norm),axis=-1))
    norm_expand = tf.expand_dims(norm_mo,axis=-1)
    newnorm = tf.concat([norm_expand,norm_expand,norm_expand],axis=-1)
    enorm = norm/newnorm

    NdotH = tf.reduce_sum(tf.multiply(enorm,ehalf_vec),axis=-1) # (None, 256, 256)
    nh_expand = tf.expand_dims(NdotH,axis=-1)
    nh = tf.concat([nh_expand,nh_expand,nh_expand],axis=-1)

    NdotL = tf.reduce_sum(tf.multiply(enorm,elight_vec),axis=-1)
    nl_expand = tf.expand_dims(NdotL,axis=-1)
    nl = tf.concat([nl_expand,nl_expand,nl_expand],axis=-1)

    NdotV = tf.reduce_sum(tf.multiply(enorm,eview_vec),axis=-1)
    nv_expand = tf.expand_dims(NdotV,axis=-1)
    nv = tf.concat([nv_expand,nv_expand,nv_expand],axis=-1)

    VdotH = tf.reduce_sum(tf.multiply(eview_vec,ehalf_vec),axis=-1)
    vh_expand = tf.expand_dims(VdotH,axis=-1)
    vh = tf.concat([vh_expand,vh_expand,vh_expand],axis=-1)

    nh = tf.maximum(nh, 1e-8)
    nv = tf.maximum(nv, 1e-8)
    nl = tf.maximum(nl, 1e-8)
    vh = tf.maximum(vh, 1e-8)

    pi = tf.constant(3.1415926)

    r2 = rough*rough
    denominator = tf.square(nh)*(tf.pow(r2,2)-1)+1+0.0001
    Norm_distrib = r2/pi/ denominator
    norm_distrib = tf.square(Norm_distrib)*pi

    k = tf.maximum(1e-8, rough * rough * 0.5)
    shade_mask1 = nl*(1-k)+k
    shade_mask2 = nv*(1-k)+k
    shade_mask = tf.compat.v1.reciprocal(shade_mask1*shade_mask2)

    F_mi = (-5.55473*vh-6.98316)*vh
    F_mi = tf.cast(F_mi,tf.float32)
    fresnel = spec+(1-spec)*tf.pow(2.0,F_mi)

    diff_scale = diff/pi

    fr = fresnel*shade_mask*norm_distrib/4 + diff_scale # 微表面模型

    ctrender_batch = fr*nl*3.14

    return ctrender_batch

def generate_vl(w=3264, h=2448):
    """ 生成点到中心光源的向量和光强图
        （默认光源位置在图片中心）

    Args:
        w (int, optional): 图片宽. Defaults to 3264.
        h (int, optional): 图片高. Defaults to 2448.

    Returns:
        eview_vec (tensor): 像素到光源的向量 (1, w, h, 3)
        intensity (tensor): intensity分布  (1, w, h, 3)
    """
    # d = w/(2*tf.tan(alpha)), alpha = 33/180*pi
    d = w/1.29876 # 高度
    view_pos = tf.constant([w/2, h/2, d], dtype=tf.float32, shape=[1,3]) # 修改这里
    view_pos = tf.expand_dims(view_pos,axis=1)

    wgrid = tf.linspace(0.0, w, w)
    hgrid = tf.linspace(0.0, h, h)
    plane_coor = tf.concat(tf.meshgrid(wgrid,hgrid,0.0),axis=2)
    planc_expand = tf.expand_dims(plane_coor,axis=0) #[1, h, w, 3]

    # view_vec
    view_pos_expand = tf.expand_dims(view_pos,axis=1)
    view_vec = view_pos_expand - planc_expand # 图像上每个点到中心光源的向量

    view_norm = tf.sqrt(tf.reduce_sum(tf.square(view_vec),axis=-1))
    view_expand = tf.expand_dims(view_norm,axis=-1)
    newview_vec = tf.concat([view_expand,view_expand,view_expand],axis=-1)
    eview_vec = view_vec/newview_vec

    # I
    xy = planc_expand[:,:,:,0:2]
    t = tf.sqrt(tf.reduce_sum(tf.square(xy - tf.constant([w/2, h/2], dtype=tf.float32, shape=[1,2])), keepdims=True, axis=-1))
    I = tf.exp(tf.square(tf.tan(t/d*1.6)) * (-0.5))
    I = tf.concat([I,I,I], axis=-1)
    return eview_vec, I