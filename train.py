# coding=utf-8
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import DA_DatasetFromFolder, TestDatasetFromFolder
import numpy as np
import random
from model.network import CDNet
from train_options import parser
import itertools
from loss.losses import cross_entropy
from collections import OrderedDict
import ever as er

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 移除手动设置设备，因为 Python 中的 CUDA 设备由 PyTorch 自动管理
# torch.cuda.set_device(3)

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

def val(best_iou):
    CDNet.eval()
    iou = 0
    with torch.no_grad():
        val_bar = tqdm(test_loader, desc="Validation")
        metric_op = er.metric.PixelMetric(2, logdir=None, logger=None)

        for hr_img1, hr_img2, label, name in val_bar:
            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.float)

            label = torch.argmax(label, 1).unsqueeze(1).float()

            cd_map, _, _ = CDNet(hr_img1, hr_img2)

            cd_map = torch.argmax(cd_map, 1).unsqueeze(1).float()

            gt_value = (label > 0).float()
            prob = (cd_map > 0).float()
            prob = prob.cpu().detach().numpy()

            gt_value = gt_value.cpu().detach().numpy()
            gt_value = np.squeeze(gt_value)
            result = np.squeeze(prob)
            metric_op.forward(gt_value, result)

        re = metric_op.summary_all()
        # 计算 IoU
        iou = re.rows[1][1]

    print(f'当前 IoU: {iou:.4f}, 最佳 IoU: {best_iou:.4f}')

    # 如果当前 IoU 优于最佳 IoU，则保存模型
    if iou > best_iou or epoch == 1:
        save_path = os.path.join(args.model_dir, f'{epoch:03}_{iou:.4f}_best.pth')
        torch.save(CDNet.state_dict(), save_path)
        print(f'模型权重已保存到: {save_path}')
        best_iou = iou
    CDNet.train()
    return best_iou

def train_epoch():
    CDNet.train()
    for hr_img1, hr_img2, label in train_bar:
        running_results['batch_sizes'] += args.batchsize

        hr_img1 = hr_img1.to(device, dtype=torch.float)
        hr_img2 = hr_img2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)
        label = torch.argmax(label, 1).unsqueeze(1).float()

        result1, result2, result3 = CDNet(hr_img1, hr_img2)

        CD_loss = CDcriterionCD(result1, label) + CDcriterionCD(result2, label) + CDcriterionCD(result3, label)

        optimizer.zero_grad()
        CD_loss.backward()
        optimizer.step()

        running_results['CD_loss'] += CD_loss.item() * args.batchsize

        train_bar.set_description(
            desc='[%d/%d] loss: %.4f' % (
                epoch, args.num_epochs,
                running_results['CD_loss'] / running_results['batch_sizes'],))


if __name__ == '__main__':
    best_iou = 0  # 初始化最佳 IoU

    # 加载数据
    train_set = DA_DatasetFromFolder(args.hr1_train, args.hr2_train, args.lab_train, crop=False)
    test_set = TestDatasetFromFolder(args, args.hr1_test, args.hr2_test, args.lab_test)

    train_loader = DataLoader(dataset=train_set, num_workers=args.num_workers, batch_size=args.batchsize, shuffle=True)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=False)



    # 定义模型
    CDNet = CDNet(img_size=args.img_size).to(device, dtype=torch.float)

    # 如果有多GPU，使用 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个 GPU 进行训练.")
        CDNet = torch.nn.DataParallel(CDNet)
    else:
        print("仅使用单个 GPU 进行训练.")

    # 设置优化器
    optimizer = optim.AdamW(itertools.chain(CDNet.parameters()), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    CDcriterionCD = cross_entropy().to(device, dtype=torch.float)

    # 训练循环
    for epoch in range(1, args.num_epochs + 1):
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        running_results = {'batch_sizes': 0, 'CD_loss': 0}
        train_epoch()
        best_iou = val(best_iou)
