import torch.nn as nn
import torch.nn.functional as F


class ResnetUnary(nn.Module):
    def __init__(self, **config):
        super(ResnetUnary, self).__init__()
        self.num_res_blocks = config.get('num_res_blocks', 7)
        self.stem_features = config.get('stem_features', 32)
        self.stem_ksize = config.get('stem_ksize', 5)
        self.stem_stride = config.get('stem_stride', 2)
        self.unary_features = config.get('unary_features', 32)
        self.unary_ksize = config.get('unary_ksize', 3)
        self.projection_features = config.get('projection_features', 32)
        self.projection_ksize = config.get('projection_ksize', 3)
        #
        F_p, F_u = self.projection_features, self.unary_features
        K_u = self.unary_ksize
        # unary features
        self.stem_conv = nn.Conv2d(3, self.stem_features, self.stem_ksize, stride=self.stem_stride, padding=2)
        self.stem_bn = nn.BatchNorm2d(self.stem_features, affine=False)
        for i in range(1, self.num_res_blocks + 1):
            for j in range(1, 4):
                setattr(self, 'conv_{}_{}'.format(i, j), nn.Conv2d(F_u, F_u, K_u, padding=1))
                setattr(self, 'bn_{}_{}'.format(i, j), nn.BatchNorm2d(F_u, affine=False))
        self.conv_projection = nn.Conv2d(F_u, F_p, self.projection_ksize, padding=1)

    def forward(self, left, right):
        unaries = {}
        for side, image in zip(['left', 'right'], [left, right]):
            # image = (2 * (image / image.max()) - 1.0)
            x = F.relu(self.stem_conv(image), inplace=True)
            for i in range(1, self.num_res_blocks + 1):
                c1, c2, c3 = tuple([getattr(self, 'conv_{}_{}'.format(i, j)) for j in range(1, 4)])
                bn_1, bn_2, bn_3 = tuple([getattr(self, 'bn_{}_{}'.format(i, j)) for j in range(1, 4)])
                c1x = F.relu(bn_1(c1(x)), inplace=True)
                c2x = F.relu(bn_2(c2(c1x)), inplace=True)
                c2x = F.relu(bn_3(c3(c2x)), inplace=True)
                x = c1x + c2x
            unaries[side] = self.conv_projection(x)
        un_l, un_r = unaries['left'], unaries['right']
        return un_l, un_r


class SeLuResblock(nn.Module):
    def __init__(self, in_feats, feats, ksize):
        super(SeLuResblock, self).__init__()
        self.conv_1 = nn.Conv2d(in_feats, feats, ksize, padding=1)
        self.conv_2 = nn.Conv2d(feats, feats, ksize, padding=1)
        self.conv_3 = nn.Conv2d(feats, feats, ksize, padding=1)

    def forward(self, x):
        c1 = F.selu(self.conv_1(x))
        c2 = F.selu(self.conv_2(c1))
        c3 = F.selu(self.conv_3(c2))
        return c1 + c3


class SeLuResnetUnary(nn.Module):
    def __init__(self, **config):
        super(SeLuResnetUnary, self).__init__()
        self.num_res_blocks = config.get('num_res_blocks', 7)
        self.stem_features = config.get('stem_features', 32)
        self.stem_ksize = config.get('stem_ksize', 5)
        self.stem_stride = config.get('stem_stride', 2)
        self.unary_features = config.get('unary_features', 32)
        self.unary_ksize = config.get('unary_ksize', 3)
        self.projection_features = config.get('projection_features', 32)
        self.projection_ksize = config.get('projection_ksize', 3)
        #
        F_p, F_u = self.projection_features, self.unary_features
        K_u = self.unary_ksize
        # unary features
        self.stem_conv = nn.Conv2d(3, self.stem_features, self.stem_ksize, stride=self.stem_stride, padding=2)
        for i in range(1, self.num_res_blocks + 1):
            setattr(self, 'resblock_{}'.format(i), SeLuResblock(F_u, F_u, K_u))
        self.conv_projection = nn.Conv2d(F_u, F_p, self.projection_ksize, padding=1)

    def forward(self, left, right):
        unaries = {}
        for side, image in zip(['left', 'right'], [left, right]):
            # image = (2 * (image / image.max()) - 1.0)
            x = F.selu(self.stem_conv(image))
            for i in range(1, self.num_res_blocks + 1):
                resblock = getattr(self, 'resblock_{}'.format(i))
                x = resblock(x)
            unaries[side] = self.conv_projection(x)
        un_l, un_r = unaries['left'], unaries['right']
        return un_l, un_r


class SeLuConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(SeLuConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x):
        return F.selu(self.conv(x))


class UnaryMCCNN(nn.Module):
    def __init__(self, **config):
        super(UnaryMCCNN, self).__init__()
        self.num_layers = config.get('num_layers', 4)
        self.unary_ksize = config.get('unary_ksize', 3)
        self.unary_features = config.get('unary_features', 32)
        self.stem_stride = config.get('stem_strides', 2)

        self.features = nn.Sequential(
            SeLuConv(3, self.unary_features, self.unary_ksize, stride=self.stem_stride, padding=1),
            *[SeLuConv(self.unary_features, self.unary_features, self.unary_ksize, padding=1) for _ in
              range(self.num_layers - 2)],
            nn.Conv2d(self.unary_features, self.unary_features, self.unary_ksize, padding=1)
        )

    def forward(self, left, right):
        un_l, un_r = self.features(left), self.features(right)
        return un_l, un_r
