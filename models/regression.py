import torch
import torch.nn as nn
import torch.nn.functional as F


class ResnetRegression(nn.Module):
    def __init__(self, **config):
        super(ResnetRegression, self).__init__()
        # configuration
        self.unary_features = config.get('unary_features', 32)
        self.unary_ksize = config.get('unary_ksize', 3)
        self.projection_features = config.get('projection_features', 32)
        self.projection_ksize = config.get('projection_ksize', 3)
        self.max_disp = config.get('max_disp', 192)
        #
        F_p, F_u = self.projection_features, self.unary_features
        K_u = self.unary_ksize
        conv_3d = {
            1: (nn.Conv3d(2 * F_p, 1 * F_u, K_u, padding=1),
                nn.Conv3d(1 * F_u, 1 * F_u, K_u, padding=1),
                nn.Conv3d(1 * F_u, 1 * F_u, K_u, padding=1, stride=2),),
            2: (nn.Conv3d(1 * F_p, 2 * F_u, K_u, padding=1),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, stride=2),),
            3: (nn.Conv3d(2 * F_p, 2 * F_u, K_u, padding=1),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, stride=2),),
            4: (nn.Conv3d(2 * F_p, 2 * F_u, K_u, padding=1),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1),
                nn.Conv3d(2 * F_u, 4 * F_u, K_u, padding=1, stride=2),),
        }
        bn_3d = {
            1: (nn.BatchNorm3d(1 * F_u),
                nn.BatchNorm3d(1 * F_u),
                nn.BatchNorm3d(1 * F_u),),
            2: (nn.BatchNorm3d(2 * F_u),
                nn.BatchNorm3d(2 * F_u),
                nn.BatchNorm3d(2 * F_u),),
            3: (nn.BatchNorm3d(2 * F_u),
                nn.BatchNorm3d(2 * F_u),
                nn.BatchNorm3d(2 * F_u),),
            4: (nn.BatchNorm3d(2 * F_u),
                nn.BatchNorm3d(2 * F_u),
                nn.BatchNorm3d(4 * F_u),),
        }
        for i in sorted(conv_3d.keys()):
            for j in range(3):
                setattr(self, 'conv_{}_{}'.format(i, j + 1), conv_3d[i][j])
                setattr(self, 'bn_{}_{}'.format(i, j + 1), bn_3d[i][j])

        self.conv_3d_b_1 = nn.Conv3d(4 * F_u, 4 * F_u, K_u, padding=1)
        self.bn_3d_b_1 = nn.BatchNorm3d(4 * F_u)
        self.conv_3d_b_2 = nn.Conv3d(4 * F_u, 4 * F_u, K_u, padding=1)
        self.bn_3d_b_2 = nn.BatchNorm3d(4 * F_u)
        pad = 1
        opad = 1

        deconv_3d = {
            4: nn.ConvTranspose3d(4 * F_u, 2 * F_u, K_u, stride=2, padding=pad, output_padding=opad),
            3: nn.ConvTranspose3d(2 * F_u, 2 * F_u, K_u, stride=2, padding=pad, output_padding=opad),
            2: nn.ConvTranspose3d(2 * F_u, 2 * F_u, K_u, stride=2, padding=pad, output_padding=opad),
            1: nn.ConvTranspose3d(2 * F_u, 1 * F_u, K_u, stride=2, padding=pad, output_padding=opad)
        }
        deconv_bn_3d = {
            4: nn.BatchNorm3d(2 * F_u),
            3: nn.BatchNorm3d(2 * F_u),
            2: nn.BatchNorm3d(2 * F_u),
            1: nn.BatchNorm3d(1 * F_u)
        }

        for i in reversed(sorted(deconv_3d.keys())):
            setattr(self, 'deconv_{}'.format(i), deconv_3d[i])
            setattr(self, 'bn_deconv_{}'.format(i), deconv_bn_3d[i])

        self.deconv_3d_proj = nn.ConvTranspose3d(F_u, 1, K_u, stride=2, padding=pad, output_padding=opad)

    def forward(self, volume):
        residual = {}
        next_in = volume
        for l in range(1, 5):
            c1, c2, c3 = tuple([getattr(self, 'conv_{}_{}'.format(l, i)) for i in range(1, 4)])
            bn_1, bn_2, bn_3 = tuple([getattr(self, 'bn_{}_{}'.format(l, i)) for i in range(1, 4)])
            l1 = F.relu(bn_1(c1(next_in)), inplace=True)
            l2 = F.relu(bn_2(c2(l1)), inplace=True)
            l3 = F.relu(bn_3(c3(l1 + l2)), inplace=True)
            residual[l] = l2
            next_in = l3
        next_in = F.relu(self.bn_3d_b_1(self.conv_3d_b_1(next_in)), inplace=True)
        next_in = F.relu(self.bn_3d_b_2(self.conv_3d_b_2(next_in)), inplace=True)
        for l in reversed(range(1, 5)):
            dc = getattr(self, 'deconv_{}'.format(l))
            bn = getattr(self, 'bn_deconv_{}'.format(l))
            deconv_out = F.relu(bn(dc(next_in)), inplace=True)
            res_in = residual[l]
            next_in = deconv_out + res_in
        projection = self.deconv_3d_proj(next_in)
        return projection
