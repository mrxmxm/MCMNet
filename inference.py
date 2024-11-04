# coding=utf-8
import os
import torch
from torch.utils.data import DataLoader
from data_utils import TestDatasetFromFolder
from model.network import CDNet
from train_options import parser
import numpy as np
from PIL import Image
import argparse
import glob

def save_prediction(map_np, save_path, color_map):
    """
    将预测的类别映射到颜色并保存为图片。
    """
    height, width = map_np.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        color_image[map_np == class_idx] = color
    Image.fromarray(color_image).save(save_path)

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # 创建测试集和数据加载器
    test_set = TestDatasetFromFolder(args, args.hr1_test, args.hr2_test, args.lab_test)
    test_loader = DataLoader(dataset=test_set, num_workers=args.num_workers, batch_size=args.val_batchsize, shuffle=False)

    # 定义并加载模型
    model = CDNet(img_size=args.img_size).to(device, dtype=torch.float)

    # 自动加载验证损失最低的模型
    model_files = glob.glob(os.path.join(args.model_dir, '*_best.pth'))
    if not model_files:
        raise FileNotFoundError("没有找到模型权重文件。")
    # 假设IoU越高越好，选择IoU最高的模型
    best_model = max(model_files, key=lambda x: float(x.split('_')[1]))
    model.load_state_dict(torch.load(best_model, map_location=device, weights_only=True), strict=False)
    print(f'加载模型权重: {best_model}')
    model.eval()

    # 定义颜色映射
    COLOR_MAP = {
        0: (0, 0, 0),  # Background
        1: (255, 255, 255),      # Building
    }

    # 创建保存目录
    pred_dir = os.path.join(args.infer_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)

    with torch.no_grad():
        for hr_img1, hr_img2, label, name in test_loader:
            hr_img1 = hr_img1.to(device, dtype=torch.float)
            hr_img2 = hr_img2.to(device, dtype=torch.float)

            # 前向传播
            cd_map, _, _ = model(hr_img1, hr_img2)
            cd_map = torch.argmax(cd_map, dim=1).squeeze(0).cpu().numpy()

            # 保存预测结果
            image_name = name[0]
            save_path = os.path.join(pred_dir, f'pred_{image_name}')
            save_prediction(cd_map, save_path, COLOR_MAP)
            print(f'保存预测结果到: {save_path}')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
