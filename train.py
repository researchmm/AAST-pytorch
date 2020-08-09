import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import tqdm
from pathlib import Path
from sampler import InfiniteSamplerWrapper

from net import Net
from datasets import TrainDataset
from util import adjust_learning_rate

cudnn.benchmark = True

def train(args):

    # Device, save and log configuration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(os.path.join(args.save_dir, args.name))
    save_dir.mkdir(exist_ok=True, parents=True)
    log_dir = Path(os.path.join(args.log_dir, args.name))
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    # Prepare datasets

    content_dataset = TrainDataset(args.content_dir, args.img_size)
    texture_dataset = TrainDataset(args.texture_dir, args.img_size, gray_only=True)
    color_dataset = TrainDataset(args.color_dir, args.img_size)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=args.n_threads))
    texture_iter = iter(data.DataLoader(
        texture_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(texture_dataset),
        num_workers=args.n_threads))
    color_iter = iter(data.DataLoader(
        color_dataset, batch_size=args.batch_size,
        sampler=InfiniteSamplerWrapper(color_dataset),
        num_workers=args.n_threads))

    # Prepare network

    network = Net(args)
    network.train()
    network.to(device)

    # Training options

    opt_L = torch.optim.Adam(network.L_path.parameters(), lr=args.lr)
    opt_AB = torch.optim.Adam(network.AB_path.parameters(), lr=args.lr)

    opts = [opt_L, opt_AB]

    # Start Training

    for i in tqdm(range(args.max_iter)):
        # S1: Adjust lr and prepare data

        adjust_learning_rate(opts, iteration_count=i, args=args)

        content_l, content_ab = [x.to(device) for x in next(content_iter)]
        texture_l = next(texture_iter).to(device)
        color_l, color_ab = [x.to(device) for x in next(color_iter)]

        # S2: Forward

        l_pred, ab_pred = network(content_l, content_ab, texture_l, color_ab)

        # S3: Calculate loss

        loss_ct, loss_t = network.ct_t_loss(l_pred, content_l, texture_l)
        loss_cr = network.cr_loss(ab_pred, color_ab)

        loss_ctw = args.content_weight * loss_ct
        loss_tw = args.texture_weight * loss_t
        loss_crw = args.color_weight * loss_cr

        loss = loss_ctw + loss_tw + loss_crw

        # S4: Backward

        for opt in opts:
            opt.zero_grad()
        loss.backward()
        for opt in opts:
            opt.step()

        # S5: Summary loss and save subnets

        writer.add_scalar('loss_content', loss_ct.item(), i + 1)
        writer.add_scalar('loss_texture', loss_t.item(), i + 1)
        writer.add_scalar('loss_color', loss_cr.item(), i + 1)

        if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
            state_dict = network.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                       'network_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()
