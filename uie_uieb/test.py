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
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


CHECKPOINTS_DIR = opt.checkpoints_dir  
INP_DIR = r"E:\seagrass_training\unet-pytorch-main\VOCdevkit\VOC2007\JPEGImagess"     # 修改為您的圖像目錄
RESULT_DIR = r"E:\seagrass_training\unet-pytorch-main\VOCdevkit\VOC2007\JPEGImages"   # 輸出目錄
model_path = r"E:\seagrass_training\Deep-WaveNet-Underwater-Image-Restoration\uie_uieb\ckpts\netG_295.pt"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

checkpoint = torch.load(model_path)

torch.serialization.add_safe_globals([argparse.Namespace])

network = CC_Module()
try:
    checkpoint = torch.load(model_path, weights_only=True)
    network.load_state_dict(checkpoint['model_state_dict'])
except KeyError:
    network.load_state_dict(checkpoint)
network.eval()
network.to(device)


if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)


class TestDataset(Dataset):
    def __init__(self, image_dir, transform=None, valid_extensions=('.jpg', '.png')):
        self.image_dir = image_dir
        self.transform = transform
        
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

            if(h > 1080 or w > 1920):
                img = F.interpolate(img, size=(1080, 1920), mode= 'bilinear', align_corners=False)

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


