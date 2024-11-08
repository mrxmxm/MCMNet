# coding=utf-8
import os
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import LoadDatasetFromFolder, DA_DatasetFromFolder, calMetric_iou,TestDatasetFromFolder
import numpy as np
import random
from model.network import CDNet
from train_options import parser
import itertools
from loss.losses import cross_entropy
from ever import opt
import time
from collections import OrderedDict
import ever as er
import cv2 as cv
import numpy as np
import logging
from PIL import Image
import matplotlib.pyplot as plt

args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
seed_torch(2022)

COLOR_MAP = OrderedDict(
    Background=(0, 0, 0),
    Building=(255, 255, 255),
)

def tes(mloss):
    CDNet.eval()
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        testing_results = {'batch_sizes': 0, 'IoU': 0}

        metric_op = er.metric.PixelMetric(2, logdir=None, logger=None)
        for hr_img1, hr_img2, label, name in test_bar:
            testing_results['batch_sizes'] += args.val_batchsize

            # print(hr_img1.shape, hr_img2.shape, label, name)

            # 将label从PyTorch张量转换为NumPy数组
            # label_np = hr_img1.squeeze().cpu().detach().numpy()

            # # 使用matplotlib显示图片
            # plt.imshow(label_np, cmap='gray')
            # plt.show()

            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)



            label = torch.argmax(label, 1).unsqueeze(1).float()

            cd_map, _, _ = CDNet(hr_img1, hr_img2)
            # 将cd_map从PyTorch张量转换为NumPy数组
            cd_map_np = cd_map.squeeze().cpu().detach().numpy()
            # print(cd_map.shape, cd_map_np.shape)

            cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()


            gt_value = (label > 0).float()
            prob = (cd_map > 0).float()
            prob = prob.cpu().detach().numpy()

            gt_value = gt_value.cpu().detach().numpy()
            gt_value = np.squeeze(gt_value)
            result = np.squeeze(prob)
            # print(result.shape, gt_value.shape)


            metric_op.forward(gt_value, result)
        re = metric_op.summary_all()

    CDNet.train()
    return mloss

import cv2 as cv
if __name__ == '__main__':
    mloss = 0

    test_set = TestDatasetFromFolder(args, args.hr1_test, args.hr2_test, args.lab_test)

    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=True)

    # define model
    CDNet = CDNet(img_size = args.img_size).to(device, dtype=torch.float)

    CDNet.load_state_dict(torch.load('/home/wu_zhiming/AMT_Net/levircd/065__0.8611_best.pth', weights_only=True), strict=False)
    mloss = tes(mloss)
