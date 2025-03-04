import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pyiqa
import torch
import os
import numpy as np
from PIL import Image
import cv2
from argparse import ArgumentParser
from time import gmtime, strftime
from tqdm import tqdm
import torchvision


# Array of dataset paths
scenes=(
    "dataset/benchmark/apples2",
    "dataset/benchmark/basketball2",
    "dataset/benchmark/bear",
    "dataset/benchmark/bear2",
    "dataset/benchmark/bin",
    "dataset/benchmark/bucket",
    "dataset/benchmark/clothes",
    "dataset/benchmark/dabao",
    "dataset/benchmark/desk3",
    "dataset/benchmark/detergent",
    "dataset/benchmark/flowerbed2",
    "dataset/benchmark/jansport",
    "dataset/benchmark/msi",
    "dataset/benchmark/nestea",
    "dataset/benchmark/pillow",
    "dataset/benchmark/rocks",
    "dataset/benchmark/sofa",
    "dataset/benchmark/sponge2",
    "dataset/benchmark/sprite",
    "dataset/benchmark/tea",
    # Add more datasets here
)
outputs=(
    "output/benchmark/apples2",
    "output/benchmark/basketball2",
    "output/benchmark/bear",
    "output/benchmark/bear2",
    "output/benchmark/bin",
    "output/benchmark/bucket",
    "output/benchmark/clothes",
    "output/benchmark/dabao",
    "output/benchmark/desk3",
    "output/benchmark/detergent",
    "output/benchmark/flowerbed2",
    "output/benchmark/jansport",
    "output/benchmark/msi",
    "output/benchmark/nestea",
    "output/benchmark/pillow",
    "output/benchmark/rocks",
    "output/benchmark/sofa",
    "output/benchmark/sponge2",
    "output/benchmark/sprite",
    "output/benchmark/tea"
    # Add more outputs here
)

img2mse = lambda x, y: np.mean((x - y) ** 2)
img2l1 = lambda x, y: np.mean(np.abs(x - y))

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        # img = Image.open(os.path.join(folder, filename))
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def load_depths_from_folder(folder):
    depths = []
    filenames = []
    for filename in sorted(os.listdir(folder)):
        depth = np.fromfile(os.path.join(folder, filename), dtype='float32')
        print('--------------depth: ', depth)
        print('--------------depth.shape: ', depth.shape)
        # depth = depth.reshape(192, 256) / 255.   
        if depth is not None:
            depths.append(depth)
            filenames.append(filename)
    return depths, filenames

def load_masks_from_folder(mask_folder):
    images = []
    filenames = []
    for filename in sorted(os.listdir(mask_folder)):
        # img = Image.open(os.path.join(mask_folder, filename))
        img = cv2.imread(os.path.join(mask_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


print(pyiqa.list_models())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 1. define required metrics
iqa_metric_psnr = pyiqa.create_metric('psnr', device=device)
iqa_metric_lpips = pyiqa.create_metric('lpips', device=device)
iqa_metric_fid = pyiqa.create_metric('fid', device=device)

# 2.Evaluation
sum_sum_score_psnr = 0.0
sum_sum_score_lpips = 0.0
sum_sum_score_fid = 0.0
# sum_sum_score_l1 = 0.0
# sum_sum_score_l2 = 0.0

save_file_path = os.path.join(f"ours_eval_{strftime('%Y-%m-%d %H:%M:%S', gmtime())}.txt")
for scene, output in zip(scenes, outputs):
    print('--------------------processing scene_', scene)
    
    predicted_folder_path = os.path.join(output, 'gs_finetuned/test/ours_7000/renders')
    GT_folder_path = os.path.join(scene, 'images_origin')
    mask_folder_path = os.path.join(scene, 'masks')

    
    sum_score_psnr = 0.0
    sum_score_lpips = 0.0
    sum_score_fid = 0.0
    # sum_score_l1 = 0.0
    # sum_score_l2 = 0.0

    predicted_images, predicted_filenames  = load_images_from_folder(predicted_folder_path)
    GT_images, GT_filenames  = load_images_from_folder(GT_folder_path)
    masks, mask_names = load_masks_from_folder(mask_folder_path)
    
    tmp_render_path = 'tmp_renders/{}'.format(scene.split('/')[-1])
    tmp_gt_path = 'tmp_gts/{}'.format(scene.split('/')[-1])
    os.makedirs(tmp_render_path, exist_ok=True)
    os.makedirs(tmp_gt_path, exist_ok=True)
    
    # psnr & lpips
    for i in tqdm(range(len(predicted_images))):
        assert os.path.splitext(predicted_filenames[i])[0] == os.path.splitext(GT_filenames[i])[0] == os.path.splitext(mask_names[i])[0]
        
        render = torch.from_numpy(predicted_images[i]).permute(2,0,1)[None]/255.
        gt = torch.from_numpy(GT_images[i]).permute(2,0,1)[None]/255.
        mask_npy = masks[i]
        
        contours, _ = cv2.findContours(mask_npy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contours[-1])
        
        render = render[:, :, y:y+h, x:x+w]
        gt = gt[:, :, y:y+h, x:x+w]
        
        torchvision.utils.save_image(render, os.path.join(tmp_render_path, predicted_filenames[i]))
        torchvision.utils.save_image(gt, os.path.join(tmp_gt_path, predicted_filenames[i]))

        score1 = iqa_metric_psnr(render, gt)
        sum_score_psnr += score1

        score2 = iqa_metric_lpips(render, gt)
        sum_score_lpips += score2
    print('---------score_psnr: ', sum_score_psnr/len(predicted_images))
    print('---------score_lpips: ', sum_score_lpips/len(predicted_images))

    # FID
    sum_score_fid = iqa_metric_fid(tmp_render_path, tmp_gt_path)
    print('---------score_fid: ', sum_score_fid)
    # save as txt
    sum_sum_score_psnr += sum_score_psnr/len(predicted_images)
    sum_sum_score_lpips += sum_score_lpips/len(predicted_images)
    sum_sum_score_fid += sum_score_fid
    # sum_sum_score_l2 += sum_score_l2/len(predicted_images)
    # sum_sum_score_l1 += sum_score_l1/len(predicted_images)
    
    # save_file_path = f'/mnt/lustre/hhchen/SPIn-NeRF-master/logs/{folder_name}/testset_010000/eval.txt'
    with open(save_file_path, 'a+') as file:
        file.write(f"{scene.split('/')[-1]}" + '\n')  # Write each value on a new line
        file.write("PSNR: " + str((sum_score_psnr/len(predicted_images)).item()) + '\n')  # Write each value on a new line
        file.write("LPIPS: " + str((sum_score_lpips/len(predicted_images)).item()) + '\n')  # Write each value on a new line
        file.write("FID: " + str(sum_score_fid) + '\n')  # Write each value on a new line
        # file.write(str(sum_score_l2) + '\n')  # Write each value on a new line
        # file.write(str(sum_score_l1) + '\n')  # Write each value on a new line

# os.remove('tmp_renders')
# os.remove('tmp_gts')
with open(save_file_path, 'a+') as file:
    print('---------sum_sum_score_psnr: ', sum_sum_score_psnr/len(scenes))
    print('---------sum_sum_score_lpips: ', sum_sum_score_lpips/len(scenes))
    print('---------sum_sum_score_fid: ', sum_sum_score_fid/len(scenes))
    
    file.write(f"{len(scenes)} scenes average metrics:" + '\n')  # Write each value on a new line
    file.write("PSNR: " + str((sum_sum_score_psnr/len(scenes)).item()) + '\n')  # Write each value on a new line
    file.write("LPIPS: " + str((sum_sum_score_lpips/len(scenes)).item()) + '\n')  # Write each value on a new line
    file.write("FID: " + str((sum_sum_score_fid/len(scenes)).item()) + '\n')  # Write each value on a new line
