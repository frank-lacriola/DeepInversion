# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import argparse
import torch
from torch import distributed, nn
import random
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torchvision import datasets, transforms

import numpy as np
import torch.cuda.amp as amp
import os
import torchvision.models as models
from utils.utils import load_model_pytorch, distributed_is_initialized

from models.segmentation_module_BiSeNet import make_model
from models.build_BiSeNet import BiSeNet
from models.segmentation_module_BiSeNet import IncrementalSegmentationBiSeNet

random.seed(0)

class CustomPooling(nn.Module):
    def __init__(self, beta=1, r_0=0, dim=(-1, -2), mode=None):
        super(CustomPooling, self).__init__()
        self.r_0 = r_0

        self.dim = dim
        self.reset_parameters()
        self.mode = mode
        # self.device = device

        if self.mode is not None:
            # make a beta for each class --> size of tensor.
            self.beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(3)), requires_grad=True)
            # self.cuda(self.device)
        else:
            self.beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(1)), requires_grad=True)

    def reset_parameters(self, beta=None, r_0=None, dim=(-1, -2)):
        if beta is not None:
            init.zeros_(self.beta)
        if r_0 is not None:
            self.r_0 = r_0
        self.dim = dim

    def forward(self, x):
        '''
        :param x (tensor): tensor of shape [bs x K x h x w]
        :return logsumexp_torch (tensor): tensor of shape [bs x K], holding class scores per class
        '''

        if self.mode is None:
            const = self.r_0 + torch.exp(self.beta)
            _, _, h, w = x.shape
            average_constant = np.log(1. / (w * h))
            const = const.to('cuda')
            mod_out = const * x
            # logsumexp_torch = 1 / const * average_constant + 1 / const * torch.logsumexp(mod_out, dim=(-1, -2))
            logsumexp_torch = (average_constant + torch.logsumexp(mod_out, dim=(-1, -2)).to('cuda')) / const
            return logsumexp_torch
        else:
            const = self.r_0 + torch.exp(self.beta)
            _, d, h, w = x.shape

            average_constant = np.log(1. / (w * h))
            # mod_out = torch.zeros(x.shape)
            self.cuda(self.device)
            mod_out0 = const[0] * x[:, 0, :, :]
            mod_out1 = const[1] * x[:, 1, :, :]
            mod_out2 = const[2] * x[:, 2, :, :]

            mod_out = torch.cat((mod_out0.unsqueeze(1), mod_out1.unsqueeze(1), mod_out2.unsqueeze(1)), dim=1)
            # logsumexp_torch = 1 / const * average_constant + 1 / const * torch.logsumexp(mod_out, dim=(-1, -2))
            logsumexp_torch = (average_constant + torch.logsumexp(mod_out, dim=(-1, -2))) / const
            return logsumexp_torch

def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # updated since we have one image
        # >> it was pred.t()
        pred = pred.t()
        # pred = pred[:,:,0,0].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)

        customPooling = CustomPooling()
        output = customPooling(output)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def run(args):
    torch.manual_seed(args.local_rank)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # until now, we only have the net -> it is not pretrained
    if args.arch_name == "resnet50v15":
        from models.resnetv15 import build_resnet
        net = build_resnet("resnet34", "classic")
    else:
        print("loading torchvision model for inversion with the name: {}".format(args.arch_name))
        # this is the teacher
        # so we need to upload here the pre trained arch on the VOC
        # net = models.__dict__["resnet50"](pretrained=False, num_classes=16)

        checkpoint_teacher = torch.load("/content/drive/MyDrive/step-0-resnet50.pth")
        # checkpoint_teacher_v2 = {}
        # print(checkpoint_teacher)
        checkpoint_teacher = checkpoint_teacher['model_state']

        head = BiSeNet("resnet50")
        body = "resnet50"
        classes_edit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        net = IncrementalSegmentationBiSeNet(body, head, classes=[16], fusion_mode="mean")

        # print(net)
        # net.supervision1[0] = nn.Conv2d(in_channels=1024, out_channels=16, kernel_size=1)
        # net.supervision2[0] = nn.Conv2d(in_channels=2048, out_channels=16, kernel_size=1)
        # net.cls[0] = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1)

        # print(net)

        net.load_state_dict(checkpoint_teacher, strict=False)

        """
        for k, v in checkpoint_teacher.items():
            # we're interested into : head.context_path

            if k.split(".")[1] == "context_path" \
                    and k.split(".")[2] not in ["features"]:
                new_k = k.replace("head.context_path.", "")
                checkpoint_teacher_v2[new_k] = v

            # edge cases -> we moved out the fc layers for BiSeNet, so
            # they are called on our checkpoint "cls"
            if k.split(".")[0] == "cls":

                if k.split(".")[2] == "bias":
                    checkpoint_teacher_v2["fc.bias"] = v
                elif k.split(".")[2] == "weight":
                    checkpoint_teacher_v2["fc.weight"] = v

        net.fc = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1)
        net.load_state_dict(checkpoint_teacher_v2)
        print(net)
        net.layer4[2].bn3 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        """

    net = net.to(device)

    use_fp16 = args.fp16


    print('==> Resuming from checkpoint..')

    ### load models
    # it it the checkpoint of the whole deepinversion model(?)
    if args.arch_name == "resnet50v15":
        path_to_model = "./models/resnet50v15/model_best.pth.tar"
        load_model_pytorch(net, path_to_model, gpu_n=torch.cuda.current_device())

    net.to(device)
    net.eval()

    # reserved to compute test accuracy on generated images by different networks
    net_verifier = None
    if args.verifier and args.adi_scale == 0:
        # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
        if args.local_rank == 0:
            print("loading verifier: ", args.verifier_arch)
            # here we should load our pre trained network on the VOC
            net_verifier = models.__dict__[args.verifier_arch](pretrained=False).to(device)
            net_verifier.eval()


    # since there is competiton among teacher and student
    # the verifier will be the actual student
    if args.adi_scale != 0.0:
        student_arch = "resnet18"
        # here we should load our pre trained network on the VOC

        """
        net_verifier = models.__dict__[student_arch](pretrained=False, num_classes=16).to(device)
        net_verifier.eval()

        checkpoint_ver = torch.load("/content/drive/MyDrive/step-0-resnet18.pth")['model_state']
        checkpoint_ver_v2 = {}

        for k, v in checkpoint_ver.items():
            # we're interested into : head.context_path

            if k.split(".")[1] == "context_path" and k.split(".")[2] != "features":
                new_k = k.replace("head.context_path.", "")
                checkpoint_ver_v2[new_k] = v

            # edge cases -> we moved out the fc layers for BiSeNet, so
            # they are called on our checkpoint "cls"
            if k.split(".")[0] == "cls":
                if k.split(".")[2] == "bias":
                    checkpoint_ver_v2["fc.bias"] = v
                elif k.split(".")[2] == "weight":
                    checkpoint_ver_v2["fc.weight"] = v
        """

        checkpoint_verifier = torch.load("/content/drive/MyDrive/step-0-resnet18.pth")
        # checkpoint_teacher_v2 = {}
        # print(checkpoint_teacher)
        checkpoint_verifier = checkpoint_verifier['model_state']
        head = BiSeNet("resnet18")
        body = "resnet18"
        classes_edit = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        net_verifier = IncrementalSegmentationBiSeNet(body, head, classes=[16], fusion_mode="mean")

        net_verifier.load_state_dict(checkpoint_verifier, strict=False)

        # checkpoint_ver_v2['fc.weight'] = checkpoint_ver_v2['fc.weight'][:, :, 0, 0]
        # net_verifier.fc = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=1)
        # net_verifier.load_state_dict(checkpoint_ver_v2)

        net_verifier = net_verifier.to(device)
        net_verifier.train()

        """if use_fp16:
            for module in net_verifier.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval().half()"""

    from deepinversion import DeepInversionClass

    exp_name = args.exp_name
    # final images will be stored here:
    adi_data_path = "./final_images/%s" % exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s" % exp_name

    args.iterations = 2000
    args.start_noise = True
    # args.detach_student = False

    args.resolution = 224
    bs = args.bs
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 256
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True

    parameters["do_flip"] = args.do_flip
    parameters["random_label"] = args.random_label
    parameters["store_best_images"] = args.store_best_images

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = args.r_feature
    coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    coefficients["tv_l1"] = args.tv_l1
    coefficients["tv_l2"] = args.tv_l2
    coefficients["l2"] = args.l2
    coefficients["lr"] = args.lr
    coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    coefficients["adi_scale"] = args.adi_scale

    network_output_function = lambda x: x

    # check accuracy of verifier
    if args.verifier:
        hook_for_display = lambda x, y: validate_one(x, y, net_verifier)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             final_data_path=adi_data_path,
                                             path=exp_name,
                                             parameters=parameters,
                                             setting_id=args.setting_id,
                                             bs=bs,
                                             use_fp16=args.fp16,
                                             jitter=jitter,
                                             criterion=criterion,
                                             coefficients=coefficients,
                                             network_output_function=network_output_function,
                                             hook_for_display=hook_for_display)
    net_student = None
    if args.adi_scale != 0:
        net_student = net_verifier
    DeepInversionEngine.generate_batch(net_student=net_student, n_batches=args.n_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nb', '--n_batches', type=int, default=1, help='Number of batches to generate for each run')
    parser.add_argument('-s', '--worldsize', type=int, default=1, help='Number of processes participating in the job.')
    parser.add_argument('--local_rank', '--rank', type=int, default=0, help='Rank of the current process.')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='Coefficient for Adaptive Deep Inversion')
    parser.add_argument('--no-cuda', action='store_true')

    parser.add_argument('--epochs', default=20000, type=int, help='batch size')
    parser.add_argument('--setting_id', default=0, type=int,
                        help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
    parser.add_argument('--bs', default=64, type=int, help='batch size')
    parser.add_argument('--jitter', default=30, type=int, help='batch size')
    parser.add_argument('--comment', default='', type=str, help='batch size')
    parser.add_argument('--arch_name', default='resnet50', type=str, help='model name from torchvision or resnet50v15')

    parser.add_argument('--fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')

    parser.add_argument('--verifier', action='store_true', help='evaluate batch with another model')
    parser.add_argument('--verifier_arch', type=str, default='mobilenet_v2',
                        help="arch name from torchvision models to act as a verifier")

    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label for optimization')
    parser.add_argument('--r_feature', type=float, default=0.05,
                        help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                        help='additional multiplier on first bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for total variation L1 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for total variation L2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0,
                        help='coefficient for the main loss in optimization')
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')

    args = parser.parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    run(args)


if __name__ == '__main__':
    main()
