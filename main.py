import argparse
from train import train
from test import test

def Args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--mode', type=str, default='train',
                        help='Train or test')
    parser.add_argument('--content_dir', type=str, required=True,
                        help='Directory path to a batch of content images')
    parser.add_argument('--texture_dir', type=str, required=True,
                        help='Directory path to a batch of texture images')
    parser.add_argument('--color_dir', type=str, required=True,
                        help='Directory path to a batch of Color images')
    parser.add_argument('--out_root', type=str, default='output/',
                        help='Root directory for outputs')
    parser.add_argument('--network', type=str, default='models/net_final.pth')
    parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')


    # training options
    parser.add_argument('--save_dir', default='./checkpoints',
                        help='Directory to save the checkpoints')
    parser.add_argument('--log_dir', default='./logs',
                        help='Directory to save the log')
    parser.add_argument('--name', default='AAST',
                        help='Name of this model')
    parser.add_argument('--n_feats', default=64, type=int,
                        help='Num of feature dimension')
    parser.add_argument('--img_size', default=256, type=int,
                        help='Size of input img')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=float, default=5e-5)
    parser.add_argument('--max_iter', type=int, default=160000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--texture_weight', type=float, default=10.0)
    parser.add_argument('--content_weight', type=float, default=1.0)
    parser.add_argument('--color_weight', type=float, default=10.0)
    parser.add_argument('--n_threads', type=int, default=16)
    parser.add_argument('--save_model_interval', type=int, default=10000)
    
    # test options
    parser.add_argument('--test_opt', type=str, default='TC',
                        help='Test options, ST(Style Transfer), T(Texture Only), C(Color Only), TC(Texture and Color), INT(Interpolation)')
    parser.add_argument('--int_num', type=int, default=4,
                        help='Interpolation num')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = Args()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
