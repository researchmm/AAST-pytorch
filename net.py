import torch.nn as nn
from function import calc_mean_std
from function import adaptive_instance_normalization as adain
import torch

from subnets.vgg import vgg
from subnets.decoder import decoder_L, decoder_AB


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        vgg.load_state_dict(torch.load(args.vgg))
        encoder = nn.Sequential(*list(vgg.children())[:31])
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.L_path = decoder_L
        self.AB_path = decoder_AB

        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_texture_loss(self, input, target):
        assert (input.size() == target.size())
        #assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def ct_t_loss(self, pred_l, content_l, texture_l):
        pred_l = pred_l.repeat(1, 3, 1, 1)
        input_feats = self.encode_with_intermediate(pred_l)
        target_ct = self.encode(content_l)
        target_t = self.encode_with_intermediate(texture_l)

        loss_ct = self.calc_content_loss(input_feats[-1], target_ct)
        loss_t = self.calc_texture_loss(input_feats[0], target_t[0])
        for i in range(1, len(input_feats) - 1):
            loss_t += self.calc_texture_loss(input_feats[i], target_t[i])

        return loss_ct, loss_t

    def cr_loss(self, pred_ab, color_ab):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zero = torch.zeros(pred_ab.shape[0], 1, pred_ab.shape[2], pred_ab.shape[3]).to(device)
        pred_ab = torch.cat([zero, pred_ab], dim=1)

        input_cr = self.encode_with_intermediate(pred_ab)
        target_cr = self.encode_with_intermediate(color_ab)

        loss_cr = self.calc_texture_loss(input_cr[0], target_cr[0])
        for i in range(1, len(input_cr) - 1):
            loss_cr += self.calc_texture_loss(input_cr[i], target_cr[i])

        return loss_cr

    def run_L_path(self, content_l, texture_l, alpha = 1.0):
        ct_l_feat = self.encode(content_l)
        t_l_feat = self.encode(texture_l)
        o_l_feat = adain(ct_l_feat, t_l_feat)
        o_l_feat = alpha *  o_l_feat + (1.0 - alpha) * ct_l_feat
        l_pred = self.L_path(o_l_feat)

        return l_pred

    def run_AB_path(self, content_ab, color_ab, alpha = 1.0):
        ct_ab_feat = self.encode(content_ab)
        cr_ab_feat = self.encode(color_ab)
        o_ab_feat = adain(ct_ab_feat, cr_ab_feat)
        o_ab_feat = alpha * o_ab_feat + (1.0 - alpha) * ct_ab_feat
        ab_pred = self.AB_path(o_ab_feat)

        return ab_pred

    def forward(self, content_l, content_ab, texture_l, color_ab, alpha_l=1.0, alpha_ab=1.0):
        ct_l_feat = self.encode(content_l)
        t_l_feat = self.encode(texture_l)

        ct_ab_feat = self.encode(content_ab)
        cr_ab_feat = self.encode(color_ab)

        o_l_feat = adain(ct_l_feat, t_l_feat)
        o_l_feat = alpha_l * o_l_feat + (1.0 - alpha_l) * ct_l_feat

        o_ab_feat = adain(ct_ab_feat, cr_ab_feat)
        o_ab_feat = alpha_ab * o_ab_feat + (1.0 - alpha_ab) * ct_ab_feat

        l_pred = self.L_path(o_l_feat)
        ab_pred = self.AB_path(o_ab_feat)

        return l_pred, ab_pred
