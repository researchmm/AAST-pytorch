import os

import torch

from tqdm import tqdm

from PIL import Image
from pathlib import Path

from net import Net
from datasets import TestDataset
from util import res_lab2rgb
import shutil

# Traditional Style Transfer
def StyleTransfer(args):
    # Device and output dir
    device = torch.device('cuda')
    out_dir = os.path.join(args.out_root, args.name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    ref_dir = os.path.join(out_dir, "ref")
    Path(ref_dir).mkdir(exist_ok=True, parents=True)

    # Prepare network

    network = Net(args)
    network.load_state_dict(torch.load(args.network))
    network.eval()
    network.to(device)

    # Prepare datasets

    content_dataset = TestDataset(args.content_dir, args.img_size)
    texture_dataset = TestDataset(args.texture_dir, args.img_size)
    LCT = len(content_dataset)
    LT = len(texture_dataset)

    # Save ref img
    for i in range(LCT):
        path = content_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "content_{}.jpg".format(i)))
    for i in range(LT):
        path = texture_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "texture_{}.jpg".format(i)))

    # Start Test
    N = LCT * LT
    print("LCT = {}, LT = {}, total output num = {}".format(LCT, LT, N))
    with tqdm(total=N) as t:
        with torch.no_grad():
            for i in range(LCT):
                for j in range(LT):
                    # S1: Prepare data and forward

                    content_l, content_ab = [x.to(device).unsqueeze(0) for x in content_dataset.__getitem__(i)]
                    texture_l, texture_ab = [x.to(device).unsqueeze(0) for x in texture_dataset.__getitem__(j)]
                    l_pred, ab_pred = network(content_l, content_ab, texture_l, texture_ab)

                    # S2: Save
                    rgb_img = res_lab2rgb(l_pred.squeeze(0), ab_pred.squeeze(0))

                    img = Image.fromarray(rgb_img)
                    name = 'ct{}_t{}_result.png'.format(i, j)
                    img.save(os.path.join(out_dir, name))

                    t.update(1)
    return None

# Transfer Texture only
def TextureOnly(args):
    # Device and output dir
    device = torch.device('cuda')
    out_dir = os.path.join(args.out_root, args.name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    ref_dir = os.path.join(out_dir, "ref")
    Path(ref_dir).mkdir(exist_ok=True, parents=True)

    # Prepare network

    network = Net(args)
    network.load_state_dict(torch.load(args.network))
    network.eval()
    network.to(device)

    # Prepare datasets

    content_dataset = TestDataset(args.content_dir, args.img_size, T_only=True)
    texture_dataset = TestDataset(args.texture_dir, args.img_size, gray_only=True)
    LCT = len(content_dataset)
    LT = len(texture_dataset)

    # Save ref img
    for i in range(LCT):
        path = content_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "content_{}.jpg".format(i)))
    for i in range(LT):
        path = texture_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "texture_{}.jpg".format(i)))

    # Start Test
    N = LCT * LT
    print("LCT = {}, LT = {}, total output num = {}".format(LCT, LT, N))
    with tqdm(total=N) as t:
        with torch.no_grad():
            for i in range(LCT):
                for j in range(LT):
                    # S1: Prepare data and forward

                    content_l, content_ab = [x.to(device).unsqueeze(0) for x in content_dataset.__getitem__(i)]
                    texture_l = texture_dataset.__getitem__(j).to(device).unsqueeze(0)
                    l_pred = network.run_L_path(content_l, texture_l)

                    # S2: Save
                    rgb_img = res_lab2rgb(l_pred.squeeze(0), content_ab.squeeze(0), T_only=True)

                    img = Image.fromarray(rgb_img)
                    name = 'ct{}_t{}_result.png'.format(i, j)
                    img.save(os.path.join(out_dir, name))

                    t.update(1)
    return None

# Transfer Color only
def ColorOnly(args):
    # Device and output dir
    device = torch.device('cuda')
    out_dir = os.path.join(args.out_root, args.name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    ref_dir = os.path.join(out_dir, "ref")
    Path(ref_dir).mkdir(exist_ok=True, parents=True)

    # Prepare network

    network = Net(args)
    network.load_state_dict(torch.load(args.network))
    network.eval()
    network.to(device)

    # Prepare datasets

    content_dataset = TestDataset(args.content_dir, args.img_size, C_only=True)
    color_dataset = TestDataset(args.color_dir, args.img_size)
    LCT = len(content_dataset)
    LCR = len(color_dataset)

    # Save ref img
    for i in range(LCT):
        path = content_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "content_{}.jpg".format(i)))
    for i in range(LCR):
        path = color_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "color_{}.jpg".format(i)))

    # Start Test
    N = LCT * LCR
    print("LCT = {}, LCR = {}, total output num = {}".format(LCT, LCR, N))
    with tqdm(total=N) as t:
        with torch.no_grad():
            for i in range(LCT):
                for k in range(LCR):
                    # S1: Prepare data and forward

                    content_l, content_ab = [x.to(device).unsqueeze(0) for x in content_dataset.__getitem__(i)]
                    color_l, color_ab = [x.to(device).unsqueeze(0) for x in color_dataset.__getitem__(k)]
                    ab_pred = network.run_AB_path(content_ab, color_ab)

                    # S2: Save
                    rgb_img = res_lab2rgb(content_l.squeeze(0), ab_pred.squeeze(0), C_only=True)

                    img = Image.fromarray(rgb_img)
                    name = 'ct{}_cr{}_result.png'.format(i, k)
                    img.save(os.path.join(out_dir, name))

                    t.update(1)

    return None

# Transfer texture and color together
def TextureAndColor(args):
    # Device and output dir
    device = torch.device('cuda')
    out_dir = os.path.join(args.out_root, args.name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    ref_dir = os.path.join(out_dir, "ref")
    Path(ref_dir).mkdir(exist_ok=True, parents=True)

    # Prepare network

    network = Net(args)
    network.load_state_dict(torch.load(args.network))
    network.eval()
    network.to(device)

    # Prepare datasets

    content_dataset = TestDataset(args.content_dir, args.img_size)
    texture_dataset = TestDataset(args.texture_dir, args.img_size, gray_only=True)
    color_dataset = TestDataset(args.color_dir, args.img_size)
    LCT = len(content_dataset)
    LT = len(texture_dataset)
    LCR = len(color_dataset)

    # Save ref img
    for i in range(LCT):
        path = content_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "content_{}.jpg".format(i)))
    for i in range(LT):
        path = texture_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "texture_{}.jpg".format(i)))
    for i in range(LCR):
        path = color_dataset.get_img_path(i)
        shutil.copy(path, os.path.join(ref_dir, "color_{}.jpg".format(i)))

    # Start Test
    N = LCT * LT * LCR
    print("LCT = {}, LT = {}, LCR = {}, total output num = {}".format(LCT, LT, LCR, N))
    with tqdm(total=N) as t:
        with torch.no_grad():
            for i in range(LCT):
                for j in range(LT):
                    for k in range(LCR):
                        # S1: Prepare data and forward

                        content_l, content_ab = [x.to(device).unsqueeze(0) for x in content_dataset.__getitem__(i)]
                        texture_l = texture_dataset.__getitem__(j).to(device).unsqueeze(0)
                        color_l, color_ab = [x.to(device).unsqueeze(0) for x in color_dataset.__getitem__(k)]
                        l_pred, ab_pred = network(content_l, content_ab, texture_l, color_ab)

                        # S2: Save
                        rgb_img = res_lab2rgb(l_pred.squeeze(0), ab_pred.squeeze(0))

                        img = Image.fromarray(rgb_img)
                        name = 'ct{}_t{}_cr{}_result.png'.format(i, j, k)
                        img.save(os.path.join(out_dir, name))

                        t.update(1)

    return None

# Trade-off between content, texture and color
def Interpolation(args):

    # Device and output dir
    device = torch.device('cuda')
    out_dir = os.path.join(args.out_root, args.name)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    ref_dir = os.path.join(out_dir, "ref")
    Path(ref_dir).mkdir(exist_ok=True, parents=True)

    # Prepare network

    network = Net(args)
    network.load_state_dict(torch.load(args.network))
    network.eval()
    network.to(device)

    # Prepare datasets

    content_dataset = TestDataset(args.content_dir, args.img_size)
    texture_dataset = TestDataset(args.texture_dir, args.img_size, gray_only=True)
    color_dataset = TestDataset(args.color_dir, args.img_size)
    LCT = len(content_dataset)
    LT = len(texture_dataset)
    LCR = len(color_dataset)

    # Save ref img
    path = content_dataset.get_img_path(0)
    shutil.copy(path, os.path.join(ref_dir, "content.jpg"))

    path = texture_dataset.get_img_path(0)
    shutil.copy(path, os.path.join(ref_dir, "texture.jpg"))

    path = color_dataset.get_img_path(0)
    shutil.copy(path, os.path.join(ref_dir, "color.jpg"))

    # Start Test
    N = args.int_num
    with tqdm(total=N * N) as t:
        with torch.no_grad():
            content_l, content_ab = [x.to(device).unsqueeze(0) for x in content_dataset.__getitem__(0)]
            texture_l = texture_dataset.__getitem__(0).to(device).unsqueeze(0)
            color_l, color_ab = [x.to(device).unsqueeze(0) for x in color_dataset.__getitem__(0)]
            for i in range(N):
                for j in range(N):
                    al = i / (N - 1)
                    aab = j / (N - 1)
                    l_pred, ab_pred = network(content_l, content_ab, texture_l, color_ab, alpha_l = al, alpha_ab = aab)

                    rgb_img = res_lab2rgb(l_pred.squeeze(0), ab_pred.squeeze(0))
                        
                    img = Image.fromarray(rgb_img)
                    name = 't{}_cr{}.png'.format(round(al, 2), round(aab, 2))
                    img.save(os.path.join(out_dir, name))
                        
                    t.update(1)
    return None
            

def test(args):
    print(args.test_opt)
    if args.test_opt == 'ST':
        StyleTransfer(args)
    elif args.test_opt == 'T':
        TextureOnly(args)
    elif args.test_opt == 'C':
        ColorOnly(args)
    elif args.test_opt == 'TC':
        TextureAndColor(args)
    elif args.test_opt == 'INT':
        Interpolation(args)


