#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .logger import setup_logger
from .model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

# CLASS INDEX ARE:
# 0: Background
# 1: Skin
# 2: Right eyebrow
# 3: Left eyebrow
# 4: Right Eye
# 5: Left Eye
# 6: ???
# 7: Right Ear
# 8: Left Ear
# 9: ???
# 10: Nose
# 11: Inner Mouth
# 12: Upper Lip
# 13: Lower Lip
# 14: Neck
# 15: ???
# 16: Shirt
# 17: Hair
# 18: ???
# 19: ???

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    # Nose, lower lip, neck, ???
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno * 255)
        # cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    # save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    background = np.zeros((512, 512))
    unknown_one = np.zeros((512, 512)) + 15
    shirt = np.zeros((512, 512)) + 16
    hair = np.zeros((512, 512)) + 17
    unknown_two = np.zeros((512, 512)) + 18
    unknown_three = np.zeros((512, 512)) + 19

    inner_mouth = np.zeros((512, 512)) + 11
    upper_lip = np.zeros((512, 512)) + 12
    lower_lip = np.zeros((512, 512)) + 13
    with torch.no_grad():
        for image_path in sorted(os.listdir(dspth)):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(f'{i}: {np.unique(parsing)}')
            parsing_face_neck = ~((parsing == background) + (parsing == shirt) + (parsing == unknown_one) + (parsing == unknown_two) + (parsing == unknown_three))
            parsing_mouth = ((parsing == upper_lip) + (parsing == lower_lip) + (parsing == inner_mouth))

            vis_parsing_maps(image, parsing_face_neck, stride=1, save_im=True, save_path=osp.join(respth, f'{image_path.split(".")[0]}_neckhead_{image_path.split(".")[1]}'))
            vis_parsing_maps(image, parsing_mouth, stride=1, save_im=True, save_path=osp.join(respth, f'{image_path.split(".")[0]}_mouth_{image_path.split(".")[1]}'))







if __name__ == "__main__":
    evaluate(dspth='/home/zll/data/CelebAMask-HQ/test-img', cp='79999_iter.pth')


