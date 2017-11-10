import torch.nn as nn
import torch.nn.functional as F


class ResnetRegression(nn.Module):
    def __init__(self, **config):
        super(ResnetRegression, self).__init__()
        # configuration
        self.features = config.get('features', 32)
        self.ksize = config.get('unary_ksize', 3)
        self.volume_features = config.get('volume_features', 64)
        self.projection_ksize = config.get('projection_ksize', 3)
        self.max_disp = config.get('max_disp', 192)
        #
        F_p, F_u = self.volume_features, self.features
        K_u = self.ksize
        conv_3d = {
            1: (nn.Conv3d(1 * F_p, 1 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(1 * F_u, 1 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(1 * F_u, 1 * F_u, K_u, padding=1, stride=2, bias=False),),
            2: (nn.Conv3d(1 * F_u, 2 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, stride=2, bias=False),),
            3: (nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, stride=2, bias=False),),
            4: (nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(2 * F_u, 2 * F_u, K_u, padding=1, bias=False),
                nn.Conv3d(2 * F_u, 4 * F_u, K_u, padding=1, stride=2, bias=False),),
        }
        bn_3d = {
            1: (nn.BatchNorm3d(1 * F_u, affine=False),
                nn.BatchNorm3d(1 * F_u, affine=False),
                nn.BatchNorm3d(1 * F_u, affine=False),),
            2: (nn.BatchNorm3d(2 * F_u, affine=False),
                nn.BatchNorm3d(2 * F_u, affine=False),
                nn.BatchNorm3d(2 * F_u, affine=False),),
            3: (nn.BatchNorm3d(2 * F_u, affine=False),
                nn.BatchNorm3d(2 * F_u, affine=False),
                nn.BatchNorm3d(2 * F_u, affine=False),),
            4: (nn.BatchNorm3d(2 * F_u, affine=False),
                nn.BatchNorm3d(2 * F_u, affine=False),
                nn.BatchNorm3d(4 * F_u, affine=False),),
        }
        for i in sorted(conv_3d.keys()):
            for j in range(3):
                setattr(self, 'conv_{}_{}'.format(i, j + 1), conv_3d[i][j])
                setattr(self, 'bn_{}_{}'.format(i, j + 1), bn_3d[i][j])

        self.conv_3d_b_1 = nn.Conv3d(4 * F_u, 4 * F_u, K_u, padding=1, bias=False)
        self.bn_3d_b_1 = nn.BatchNorm3d(4 * F_u, affine=False)
        self.conv_3d_b_2 = nn.Conv3d(4 * F_u, 4 * F_u, K_u, padding=1)
        self.bn_3d_b_2 = nn.BatchNorm3d(4 * F_u, affine=False)
        pad = 1
        opad = 1

        deconv_3d = {
            4: nn.ConvTranspose3d(4 * F_u, 2 * F_u, K_u, stride=2, padding=pad, output_padding=opad, bias=False),
            3: nn.ConvTranspose3d(2 * F_u, 2 * F_u, K_u, stride=2, padding=pad, output_padding=opad, bias=False),
            2: nn.ConvTranspose3d(2 * F_u, 2 * F_u, K_u, stride=2, padding=pad, output_padding=opad, bias=False),
            1: nn.ConvTranspose3d(2 * F_u, 1 * F_u, K_u, stride=2, padding=pad, output_padding=opad, bias=False)
        }
        deconv_bn_3d = {
            4: nn.BatchNorm3d(2 * F_u, affine=False),
            3: nn.BatchNorm3d(2 * F_u, affine=False),
            2: nn.BatchNorm3d(2 * F_u, affine=False),
            1: nn.BatchNorm3d(1 * F_u, affine=False)
        }

        for i in reversed(sorted(deconv_3d.keys())):
            setattr(self, 'deconv_{}'.format(i), deconv_3d[i])
            setattr(self, 'bn_deconv_{}'.format(i), deconv_bn_3d[i])

        self.deconv_3d_proj = nn.ConvTranspose3d(F_u, 1, K_u, stride=2, padding=pad, output_padding=opad, bias=False)

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


class SeLuRegressionBlock(nn.Module):
    def __init__(self, feats_in, feats, feats_out, ksize):
        super(SeLuRegressionBlock, self).__init__()
        self.conv_1 = nn.Conv3d(feats_in, feats, ksize, padding=1, bias=False)
        self.conv_2 = nn.Conv3d(feats, feats, ksize, padding=1, bias=False)
        self.conv_3 = nn.Conv3d(feats, feats_out, ksize, padding=1, stride=2, bias=False)

    def forward(self, x):
        c1 = F.selu(self.conv_1(x), inplace=True)
        c2 = F.selu(self.conv_2(c1), inplace=True)
        c3 = F.selu(self.conv_3(c1 + c2), inplace=True)
        return c3, c2


class SeLuUpblock(nn.Module):
    def __init__(self, feats_in, feats_out, ksize, pad, opad):
        super(SeLuUpblock, self).__init__()
        self.conv = nn.ConvTranspose3d(feats_in, feats_out, ksize, stride=2, padding=pad, output_padding=opad,
                                       bias=False)

    def forward(self, x, skip):
        deconv = F.selu(self.conv(x), inplace=True)
        return deconv + skip


class SeLuResnetRegression(nn.Module):
    def __init__(self, **config):
        super(SeLuResnetRegression, self).__init__()
        # configuration
        self.features = config.get('features', 32)
        self.ksize = config.get('unary_ksize', 3)
        self.volume_features = config.get('volume_features', 64)
        self.projection_ksize = config.get('projection_ksize', 3)
        self.max_disp = config.get('max_disp', 192)
        #
        F_p, F_u = self.volume_features, self.features
        K_u = self.ksize

        per_block_params = [
            {'feats_in': 1 * F_p, 'feats': 1 * F_u, 'feats_out': 1 * F_u, 'ksize': K_u},
            {'feats_in': 1 * F_u, 'feats': 2 * F_u, 'feats_out': 2 * F_u, 'ksize': K_u},
            {'feats_in': 2 * F_u, 'feats': 2 * F_u, 'feats_out': 2 * F_u, 'ksize': K_u},
            {'feats_in': 2 * F_u, 'feats': 2 * F_u, 'feats_out': 4 * F_u, 'ksize': K_u},
        ]

        for i, params in enumerate(per_block_params):
            setattr(self, 'resblock_{}'.format(i + 1), SeLuRegressionBlock(**params))

        self.conv_3d_b_1 = nn.Conv3d(4 * F_u, 4 * F_u, K_u, padding=1, bias=False)
        self.conv_3d_b_2 = nn.Conv3d(4 * F_u, 4 * F_u, K_u, padding=1)
        pad = 1
        opad = 1

        deconv_params = [
            {'feats_in': 2 * F_u, 'feats_out': 1 * F_u, 'ksize': K_u, 'pad': pad, 'opad': opad},
            {'feats_in': 2 * F_u, 'feats_out': 2 * F_u, 'ksize': K_u, 'pad': pad, 'opad': opad},
            {'feats_in': 2 * F_u, 'feats_out': 2 * F_u, 'ksize': K_u, 'pad': pad, 'opad': opad},
            {'feats_in': 4 * F_u, 'feats_out': 2 * F_u, 'ksize': K_u, 'pad': pad, 'opad': opad},
        ]

        for i, params in enumerate(deconv_params):
            setattr(self, 'deconv_{}'.format(i + 1), SeLuUpblock(**params))

        self.deconv_3d_proj = nn.ConvTranspose3d(F_u, 1, K_u, stride=2, padding=pad, output_padding=opad, bias=False)

    def forward(self, volume):
        residual = {}
        next_in = volume
        for l in range(1, 5):
            block = getattr(self, 'resblock_{}'.format(l))
            out, skip = block(next_in)
            residual[l] = skip
            next_in = out
        next_in = F.selu(self.conv_3d_b_1(next_in), inplace=True)
        next_in = F.selu(self.conv_3d_b_2(next_in), inplace=True)
        for l in reversed(range(1, 5)):
            dc = getattr(self, 'deconv_{}'.format(l))
            next_in = dc(next_in, residual[l])
        projection = self.deconv_3d_proj(next_in)
        return projection