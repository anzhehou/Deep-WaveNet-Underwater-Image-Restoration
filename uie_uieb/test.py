from models import CC_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
import shutil
from tqdm import tqdm
import torchvision.transforms as T
from options import opt
import argparse

# 設置路徑
CHECKPOINTS_DIR = opt.checkpoints_dir  # 檢查點目錄，例如 '/content/Deep-WaveNet-Underwater-Image-Restoration/checkpoints'
INP_DIR = 'uie_uieb/sample'  # 修改為您的圖像目錄
RESULT_DIR = 'uie_uieb\result'  # 輸出目錄

# 設置設備
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 允許 argparse.Namespace 以修復 weights_only=True 錯誤
torch.serialization.add_safe_globals([argparse.Namespace])

# 初始化模型
network = CC_Module()
try:
    checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, "netG_295.pt"), weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
except KeyError:
    network.load_state_dict(checkpoint)
network.eval()
network.to(device)

# 創建輸出目錄
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# 定義測試數據集
class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None, valid_extensions=('.jpg', '.png')):
        self.image_dir = image_dir
        self.transform = transform
        # 支援自訂圖像擴展名
        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"圖像 {img_path} 不存在")
        img = img[:, :, ::-1]  # BGR 轉 RGB
        img = np.float32(img) / 255.0  # 標準化至 [0, 1]
        sample = {'image': img, 'name': img_name}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

# 定義變換
transforms = T.Compose([
    T.ToTensor()  # 轉為張量並標準化
])

# 主程式
if __name__ == '__main__':
    # 創建數據集和數據載入器
    test_dataset = TestDataset(
        image_dir=INP_DIR,
        transform=transforms,
        valid_extensions=('.jpg', '.png', '.jpeg', '.bmp')  # 可根據需要擴展
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 記錄開始時間
    start_time = time.time()

    # 使用 tqdm 追蹤進度
    with tqdm(total=len(test_dataset)) as t:
        for batch in test_loader:
            img = batch['image'].to(device)  # 形狀：(1, 3, H, W)
            img_name = batch['name'][0]

            # 獲取圖像尺寸
            h, w = img.shape[2], img.shape[3]

            # 模型推理
            with torch.no_grad():
                output = network(img)

            # 處理輸出
            output = (output.clamp_(0.0, 1.0)[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = output[:, :, ::-1]  # RGB 轉 BGR

            # 保存結果
            cv2.imwrite(os.path.join(RESULT_DIR, img_name), output)

            # 更新進度條
            t.set_postfix_str(f"名稱: {img_name} | 原尺寸 [高/寬]: {h}/{w} | 新尺寸 [高/寬]: {output.shape[0]}/{output.shape[1]}")
            t.update(1)

    # 計算並打印時間
    end_time = time.time()
    total_time = end_time - start_time
    print(f'總耗時（秒）：{total_time}')
    print(f'每張圖像平均耗時（秒）：{total_time / len(test_dataset)}')

