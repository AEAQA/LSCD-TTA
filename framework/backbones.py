import torch
import torch.nn as nn
from torchvision.models import AlexNet
from torchvision.models import resnet50, resnet101, alexnet
from torchvision.models import vit_b_16
import torch.nn.functional as F

__all__ = ['AlexNet', 'Resnet', 'Vit']

from framework.registry import Backbones
from models.DomainAdaptor import AdaMixBN


def init_classifier(fc):
    nn.init.xavier_uniform_(fc.weight, .1)
    nn.init.constant_(fc.bias, 0.)
    return fc


@Backbones.register('resnet101')
@Backbones.register('resnet50')
@Backbones.register('resnet18')
class Resnet(nn.Module):
    def __init__(self, num_classes, pretrained=False, args=None):
        super(Resnet, self).__init__()
        if '50' in args.backbone:
            print('Using resnet-50')
            resnet = resnet50(pretrained=pretrained)
            self.in_ch = 2048
        elif '101' in args.backbone:
            resnet = resnet101(pretrained=pretrained)
            self.in_ch = 2048
        else:
            raise ValueError('Invalid Resnet Setting!')
        
        self.conv1 = resnet.conv1
        self.relu = resnet.relu
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(self.in_ch, num_classes, bias=False)
        if args.in_ch != 3:
            self.init_conv1(args.in_ch, pretrained)

    def init_conv1(self, in_ch, pretrained):
        model_inplanes = 64
        conv1 = nn.Conv2d(in_ch, model_inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        old_weights = self.conv1.weight.data
        if pretrained:
            for i in range(in_ch):
                self.conv1.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]
        self.conv1 = conv1

    def forward(self, x):
        net = self
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)

        l1 = net.layer1(x)
        l2 = net.layer2(l1)
        l3 = net.layer3(l2)
        l4 = net.layer4(l3)
        logits = self.fc(l4.mean((2, 3)))
        return x, l1, l2, l3, l4, logits

    def get_lr(self, fc_weight):
        lrs = [
            ([self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4], 1.0),
            (self.fc, fc_weight)
        ]
        return lrs


@Backbones.register('alexnet')
class Alexnet(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(Alexnet, self).__init__()
        self.args = args
        cur_alexnet = alexnet(pretrained=pretrained)
        self.features = cur_alexnet.features
        self.avgpool = cur_alexnet.avgpool
        self.feature_layers = nn.Sequential(*list(cur_alexnet.classifier.children())[:-1])
        self.in_ch = cur_alexnet.classifier[-1].in_features
        self.fc = nn.Linear(self.in_ch, num_classes, bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        feats = self.feature_layers(x)
        output_class = self.fc(feats)
        return feats, output_class

    def get_lr(self, fc_weight):
        return [([self.features, self.feature_layers], 1.0), (self.fc, fc_weight)]


class Convolution(nn.Module):

    def __init__(self, c_in, c_out, mixbn=False):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        if mixbn:
            self.bn = AdaMixBN(c_out)
        else:
            self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(True)
        self.seq = nn.Sequential(
            self.conv,
            self.bn,
            self.relu
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


@Backbones.register('convnet')
class ConvNet(nn.Module):
    def __init__(self, num_classes, pretrained=True, args=None):
        super(ConvNet, self).__init__()

        c_hidden = 64
        mix = True
        self.conv1 = Convolution(3, c_hidden, mixbn=mix)
        self.conv2 = Convolution(c_hidden, c_hidden, mixbn=mix)
        self.conv3 = Convolution(c_hidden, c_hidden, mixbn=mix)
        self.conv4 = Convolution(c_hidden, c_hidden, mixbn=mix)

        self._out_features = 2**2 * c_hidden
        self.in_ch = 3
        self.fc = nn.Linear(self._out_features, num_classes)

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert (H == 32 and W == 32), "Input to network must be 32x32, " "but got {}x{}".format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        feat = x
        x = x.view(x.size(0), -1)
        return x[:, :, None, None], self.fc(x)

    def get_lr(self, fc_weight):
        return [(self, 1.0)]


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(LearnedPositionalEmbedding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self):
        return self.pos_embed


@Backbones.register('vit')
class Vit(nn.Module):
    def __init__(self, num_classes, pretrained=False, args=None):
        super(Vit, self).__init__()

        self.vit = vit_b_16(pretrained=pretrained)
        self.in_ch = self.vit.hidden_dim  
        self.fc = nn.Linear(self.in_ch, num_classes, bias=False)
        
        if getattr(args, 'in_ch', 3) != 3:
            self.init_patch_embed(args.in_ch, pretrained)
            
        num_patches = self.vit.encoder.pos_embedding.shape[1]
        embed_dim = self.vit.encoder.pos_embedding.shape[2]
        self.vit.encoder.pos_embedding_module = LearnedPositionalEmbedding(num_patches, embed_dim)

        with torch.no_grad():
            self.vit.encoder.pos_embedding_module.pos_embed.copy_(self.vit.encoder.pos_embedding)
        del self.vit.encoder.pos_embedding

    def init_patch_embed(self, in_ch, pretrained):
        old_conv = self.vit.conv_proj
        new_conv = nn.Conv2d(
            in_ch,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        if pretrained:
            old_weights = old_conv.weight.data
            for i in range(in_ch):
                new_conv.weight.data[:, i, :, :] = old_weights[:, i % 3, :, :]
        self.vit.conv_proj = new_conv

    def forward(self, x):
        B = x.size(0)
        x_patch = self.vit.conv_proj(x)
        out_x = x_patch
        
        B, C, H, W = x_patch.shape
        x_tokens = x_patch.flatten(2).transpose(1, 2)
        pos_embed = self.vit.encoder.pos_embedding_module()
        pos_embed = pos_embed[:, 1:, :]
        x_tokens = x_tokens + pos_embed

        blocks = self.vit.encoder.layers
        depth = len(blocks)
        group_size = depth // 4 if depth >= 4 else 1
        token = x_tokens
        l1 = l2 = l3 = None
        for i, block in enumerate(blocks):
            token = block(token)
            if (i + 1) == group_size:
                l1 = token.transpose(1, 2).reshape(B, C, H, W)
            elif (i + 1) == group_size * 2:
                l2 = token.transpose(1, 2).reshape(B, C, H, W)
            elif (i + 1) == group_size * 3:
                l3 = token.transpose(1, 2).reshape(B, C, H, W)
        l4 = token.transpose(1, 2).reshape(B, C, H, W)
        
        logits = self.fc(l4.mean(dim=(2, 3)))
        return out_x, l1, l2, l3, l4, logits

    def get_lr(self, fc_weight):
        backbone_params = [self.vit.conv_proj, self.vit.encoder.pos_embedding_module, self.vit.encoder.layers]
        return [(backbone_params, 1.0), (self.fc, fc_weight)]