from skimage.color import rgb2lab, lab2rgb
import numpy as np
import torch

def adjust_learning_rate(opts, iteration_count, args):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for opt in opts:
        for param_group in opt.param_groups:
            param_group['lr'] = lr

def my_rgb2lab(rgb_image):
    rgb_image = np.transpose(rgb_image, (1,2,0))
    lab_image = rgb2lab(rgb_image)
    l_image = np.transpose(lab_image[:,:,:1], (2,0,1))
    ab_image = np.transpose(lab_image[:,:,1:], (2,0,1))
    return l_image, ab_image

def my_lab2rgb(lab_image):
    lab_image = np.transpose(lab_image, (1,2,0))
    rgb_image = lab2rgb(lab_image)
    rgb_image = np.transpose(rgb_image, (2,0,1))
    return rgb_image

def res_lab2rgb(l, ab, T_only = False, C_only = False):
    l = l.cpu().numpy()
    ab = ab.cpu().numpy()
    a = ab[0:1]
    b = ab[1:2]

    if not C_only:
        l = l * (100.0 + 0.0) - 0.0
    if not T_only:
        a = ab[0:1] * (98.0  + 86.0) - 86.0
        b = ab[1:2] * (94.0 + 107.0) - 107.0

    lab = np.concatenate((l, a, b), axis=0)
    lab = np.transpose(lab, (1, 2, 0))
    rgb = lab2rgb(lab)
    rgb = (np.array(rgb) * 255).astype(np.uint8)
    return rgb

