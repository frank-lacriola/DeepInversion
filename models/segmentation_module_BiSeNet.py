import torch
import torch.nn as nn

import torch.nn.functional as functional

from functools import reduce

from models.build_BiSeNet import  BiSeNet

def make_model(opts, classes=None):

    # string with the backbone e.g. 'resnet'
    body = opts.backbone

    if not opts.no_pretrained:
        pretrained_path = f'pretrained/{opts.backbone}_{opts.norm_act}.pth.tar'
        pre_dict = torch.load(pretrained_path, map_location='cpu')
        del pre_dict['state_dict']['classifier.fc.weight']
        del pre_dict['state_dict']['classifier.fc.bias']

        body.load_state_dict(pre_dict['state_dict'])
        del pre_dict  # free memory

    head = BiSeNet(body)

    if classes is not None:
        model = IncrementalSegmentationBiSeNet(body, head, classes=classes, fusion_mode=opts.fusion_mode)
    else:
        # model = SegmentationModule(body, head, head_channels, opts.num_classes, opts.fusion_mode)
        pass

    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


class IncrementalSegmentationBiSeNet(nn.Module):

    def __init__(self, body, head, classes, ncm=False, fusion_mode="mean"):
        super(IncrementalSegmentationBiSeNet, self).__init__()

        self.body = body
        self.head = head

        assert isinstance(classes, list), \
            "Classes must be a list where to every index correspond the num of classes for that task"

        if body == "resnet18":
          in_channels1, in_channels2 = 256, 512
        elif body == "resnet50":
          in_channels1, in_channels2 = 1024, 2048


        # classifiers supervision 1
        self.supervision1 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels1, out_channels=c, kernel_size=1) for c in classes]
        )

        # classifiers supervision 2
        self.supervision2 = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels2, out_channels=c, kernel_size=1) for c in classes]
        )

        # classifiers for the final layers
        self.cls = nn.ModuleList(
            [nn.Conv2d(in_channels=256, out_channels=c, kernel_size=1) for c in classes]
            # [nn.Conv2d(256, c, 1) for c in classes]
        )

        self.classes = classes
        self.head_channels = 256
        self.tot_classes = reduce(lambda a, b: a + b, self.classes)
        self.means = None

    def _network(self, x, ret_intermediate=False):

        features, features_cx1, features_cx2 = self.head(x)
        self.features = features

        out = []
        cx1_out = []
        cx2_out = []

        print(self.cls)

        for mod in self.cls:
            out.append(mod(features))

        for mod in self.supervision1:
            cx1_out.append(mod(features_cx1))

        for mod in self.supervision2:
            cx2_out.append(mod(features_cx2))

        x_o = torch.cat(out, dim=1)

        cx1_sup = torch.cat(cx1_out, dim=1)
        cx2_sup = torch.cat(cx2_out, dim=1)

        # it is forced to True at the moment
        if ret_intermediate:
            return x_o, cx1_sup, cx2_sup

        return x_o

    def init_new_classifier(self, device):
        cls = self.cls[-1]

        imprinting_w = self.cls[0].weight[0]
        bkg_bias = self.cls[0].bias[0]

        bias_diff = torch.log(torch.FloatTensor([self.classes[-1] + 1])).to(device)

        new_bias = (bkg_bias - bias_diff)

        cls.weight.data.copy_(imprinting_w)
        cls.bias.data.copy_(new_bias)

        self.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def forward(self, x, scales=None, do_flip=False, ret_intermediate=False):
        out_size = x.shape[-2:]

        print("out_size", out_size)

        if ret_intermediate:
            out, out_cx1, out_cx2 = self._network(x, ret_intermediate)
            # scale factor requested for BiSeNet
            out = functional.interpolate(out, scale_factor=8, mode="bilinear", align_corners=False)
            out_1 = functional.interpolate(out_cx1, size=out_size, mode='bilinear', align_corners=False)
            out_2 = functional.interpolate(out_cx2, size=out_size, mode='bilinear', align_corners=False)
            return out, out_1, out_2
        print("X : ",x.shape)
        net_w = self._network(x, ret_intermediate)

        print("self._network : ", net_w.shape)
        f = functional.interpolate(net_w, size=out_size, mode="bilinear", align_corners=False)

        return f

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
