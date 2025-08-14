import os
import json
import pickle
import torch
import copy
import cv2

import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm
from torchvision.utils import save_image
from src.tokenizer import DiagonalGaussianDistribution
import random

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)    

def show_image(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class AugmentDegraded():
    def __init__(self, seed=12345):
        self.random_generator = np.random.default_rng(seed)
        self.resize_256 = A.Resize(256, 256)
        self.norm_torch = A.Compose([
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.ToTensorV2(),
        ])
        # self.aug_sr = A.Compose([
        #     A.OneOf([
        #         A.Resize(200,200),
        #         A.Resize(220,220),
        #     ], p=1.0),
        #     A.Resize(256, 256),
        # ])

        self.aug_noise =  A.Compose([
            # A.Resize(256, 256),
            A.OneOf([
                A.GaussNoise(std_range=(10/255.0, 10/255.0), p=1.0),
                A.GaussNoise(std_range=(15/255.0, 15/255.0), p=1.0),
                A.GaussNoise(std_range=(25/255.0, 25/255.0), p=1.0),
                A.GaussNoise(std_range=(35/255.0, 35/255.0), p=1.0),
                # A.GaussNoise(std_range=(50/255.0, 50/255.0), p=1.0),
            ], p=1.0),
        ])

        self.aug_compress = A.Compose([
            # A.Resize(256, 256),
            A.ImageCompression(quality_range=(10, 40), compression_type="jpeg", p=1.0),
        ])

        self.aug_blur = A.Compose([
            # A.Resize(256, 256),
            A.OneOf([
                A.Blur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 7), p=1.0)
            ],p=1.0),
        ])

        # self.aug_rain = A.Compose([
        #     A.Resize(256, 256),
        #     A.OneOf([
        #         A.RandomRain(drop_length=30, blur_value=1, drop_color=[255,255,255], rain_type="drizzle", brightness_coefficient=1.0, p=1.0),
        #         A.RandomRain(drop_length=25, blur_value=1, drop_color=[255,255,255], rain_type="heavy", brightness_coefficient=1.0, p=1.0),
        #     ], p=1.0),
        #     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     A.ToTensorV2(),
        # ])
        

        # self.aug_fog = A.Compose([
        #     A.Resize(256, 256),
        #     A.RandomFog(fog_coef_range=[0.8,1.0], alpha_coef=0.3, p=1.0),
        #     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     A.ToTensorV2(),
        # ])

        self.aug_fog = iaa.CloudLayer(
            intensity_mean=(220, 255),
            intensity_freq_exponent=(-2.0, -1.5),
            intensity_coarse_scale=2,
            alpha_min=(0.7, 0.9),
            alpha_multiplier=0.3,
            alpha_size_px_max=(2, 8),
            alpha_freq_exponent=(-4.0, -2.0),
            sparsity=0.9,
            density_multiplier=(0.25, 0.6),
            seed=None, name=None
        )

        self.aug_rain = iaa.Rain(
            nb_iterations=(3, 5),
            drop_size=(0.1, 0.25),
            speed=(0.04, 0.20),
        )
        self.cap_degraded = {
            "blur": [
                "This image appears to suffer from blurriness, lack of sharpness and detail, makes it difficult to discern finer elements",
                "The image appears to have some blurring or artifacting, likely from compression or post-processing.",
                "The image has visible degradation, including noticeable pixelation and blurring, reducing its sharpness and clarity.",
                "Blurry visuals obscure critical information, making the image unusable for precise analysis",
                "The overall image looks blurry, affecting clarity.",
                "Blurriness is the most evident distortion, reducing sharpness.",
                "The image suffers from a strong blur, making objects hard to distinguish.",
                "A significant blur is present, affecting all regions of the image.",
                "Motion blur is apparent, smearing details across the image.",
                "The image lacks fine details due to excessive blurring.",
                "A high level of blur is present, making edges less distinct.",
                "The image appears soft due to a lack of sharpness.",
                "Blurriness reduces the visibility of small details.",
                "The image is heavily blurred, making recognition difficult.",
            ],
            "compress": [
                "This image is affected by compression artifacts, cause a blocky, distorted appearance.",
                "The image appears to have some artifacting, likely from compression.",
                "The image likely has visible blocky or blurry regions due to heavy compression, which reduces clarity.",
                "The most noticeable issue is compression artifacts, especially in textured areas.",
                "Blocky compression artifacts are evident, lowering visual quality.",
                "The image is affected by compression artifacts, reducing smoothness.",
                "Visible compression noise is present, particularly in uniform regions.",
                "The image quality is degraded due to noticeable compression blocks.",
                "Compression artifacts are highly visible, leading to unnatural edges.",
                "The image contains blocking artifacts, impacting overall sharpness.",
                "Distinct compression patterns are apparent, reducing fidelity.",
                "The compression artifacts create unnatural transitions in color and texture.",
                "High compression causes loss of detail and unnatural pixelation."
            ],
            "noise": [
                "This image is degraded by noise, appears to have a significant amount of visual noise, which creates a speckled effect",
                "The image appears heavily degraded with visual noise, resembling a pixelated or grainy texture.",
                "The image is affected by noticeable noise.",
                "The most obvious distortion is noise, which affects the entire image.",
                "The image is dominated by noticeable noise, reducing clarity.",
                "High levels of noise are present, degrading the overall quality.",
                "The image appears grainy due to significant noise.",
                "Noise is the primary artifact, making finer details hard to discern.",
                "The presence of noise reduces the sharpness and contrast of the image.",
                "The image suffers from heavy noise, particularly in darker regions.",
                "The visual quality is affected by a high amount of noise.",
                "The image is corrupted by excessive noise, making it difficult to analyze.",
                "The image appears speckled due to prominent noise artifacts."
            ],
            "rain": [
                "There is an overlay effect that resembles rain, and it affects the clarity of the image.",
                "The image is artificially degraded with visible vertical streaks resembling rain or scratch marks.",
                "The image has noticeable degradation due to streaks resembling rain",
                "The image is affected by rain streaks, reducing visibility.",
                "Raindrops on the image create distortion, making details unclear.",
                "The scene is obscured by rain streaks, impacting clarity.",
                "Rain artifacts make the image appear streaked and noisy.",
                "The image is degraded by rainfall, causing local distortions.",
                "Raindrops create blurring effects, reducing sharpness.",
                "The rain effect results in reduced visibility of objects.",
                "Water droplets introduce random distortions across the image.",
                "The image appears wet and smeared due to rain artifacts.",
                "Raindrop reflections cause irregular brightness variations."
            ],
            "sr": [
                "This image appears to be low resolution, makes it lacks sharpness and difficult to discern fine details",
                "This image lacks fine details, makes textures appear blurry or pixelated rather than sharp",
                "This image lacks of fine detail and noticeable pixelation.",
                "The image appears blocky, with visible square pixels due to insufficient resolution.",
                "The image is composed of large, visible square pixels, reducing clarity."
                
            ],
            "fog": [
                "This image appears to be a hazy or foggy",
                "This image has a washed-out appearance, likely due to haze",
                "This image has a washed-out appearance, likely due to fog",
                "The image is covered in a layer of haze, reducing contrast.",
                "A thick fog obscures details, making the scene appear faded.",
                "The most obvious issue is haze, making objects less distinguishable.",
                "The image suffers from low contrast due to foggy conditions.",
                "Atmospheric haze makes distant objects appear unclear.",
                "The fog reduces the sharpness of edges and fine details.",
                "The image has a misty appearance, softening the entire scene.",
                "The foggy conditions result in a washed-out look.",
                "Visibility is reduced due to the presence of heavy haze.",
                "The image appears blurry and faded due to atmospheric fog."
            ]
        }
        self.map_aug = {
            "blur": self.aug_blur,
            "compress": self.aug_compress,
            "noise": self.aug_noise,
            "rain": self.aug_rain,
            "fog": self.aug_fog
        }
    
    def aug_sr(self, img):
        d_size = np.random.randint(240, 256)
        aug_downsample = A.Resize(d_size, d_size, cv2.INTER_CUBIC)
        img_downsample = aug_downsample(image=img)["image"]
        w = np.random.randint(int(img_downsample.shape[0] * 0.45), img_downsample.shape[0]-1)
        h = np.random.randint(int(img_downsample.shape[1] * 0.45), img_downsample.shape[1]-1)
        x = np.random.randint(0, img_downsample.shape[0] - w + 1)
        y = np.random.randint(0, img_downsample.shape[1] - h + 1)

        xh0 = int((x/img_downsample.shape[0]) * img.shape[0])
        xh1 = int(((x+w)/img_downsample.shape[0]) * img.shape[0])
        yh0 = int((y/img_downsample.shape[1]) * img.shape[1])
        yh1 = int(((y+h)/img_downsample.shape[1]) * img.shape[1])

        img_h = img[xh0:xh1, yh0:yh1, :]
        img_d = img_downsample[x:x+w, y:y+h, :]

        aug_upsample = A.Resize(256, 256, interpolation=np.random.randint(0, 3))
        img_d = aug_upsample(image=img_d)["image"]
        img_h = aug_upsample(image=img_h)["image"]
        img_d = self.norm_torch(image=img_d)["image"]
        img_h = self.norm_torch(image=img_h)["image"]
        return img_d, img_h

    def __call__(self, img, crop_p, aug_list = [], p = [], n_iter=1):
        if len(aug_list) == 0 or n_iter == 0:
            img_256 = self.resize_256(image=img)["image"]
            # img_resize = self.norm_torch(image=img_256)["image"]
            img_resize = self.norm_torch(image=img_256)["image"]
            return img_resize, img_resize, None, None
        
        name = np.random.choice(aug_list, size=n_iter, p=p, replace=False)[0]
        img_degraded = None
        if name == "sr":
            # print("----------------------------- SR SR")
            img_d, img_h = self.aug_sr(img=img)
            cap = self.random_generator.choice(len(self.cap_degraded[name]))
            cap = self.cap_degraded[name][cap]
            crop = True
            return img_h, img_d, [cap], [name], crop
        else:
            img_256 = self.resize_256(image=img)["image"]

            if name in ["rain", "fog"]:
                t = self.map_aug[name]
                if img_degraded is None:
                    img_degraded = t(image=img_256)
                else:
                    img_degraded = t(image=img_degraded)
                # img_degraded = self.norm_torch(image=img_d)["image"]
            else:
                t = self.map_aug[name]
                if img_degraded is None:
                    img_degraded = t(image=img_256)["image"]
                else:
                    img_degraded = t(image=img_degraded)["image"]
            cap = self.random_generator.choice(len(self.cap_degraded[name]))
            cap = self.cap_degraded[name][cap]
            if np.random.rand() < crop_p:
                img = img_256
                # crop 
                w = np.random.randint(int(img_degraded.shape[0] * 0.2), int(img_degraded.shape[0] * 0.5))
                h = np.random.randint(int(img_degraded.shape[1] * 0.2), int(img_degraded.shape[0] * 0.5))
                # print(w,h,img_degraded.shape[0],img_degraded.shape[1])
                x = np.random.randint(0, img_degraded.shape[0] - w + 1)
                y = np.random.randint(0, img_degraded.shape[1] - h + 1)

                xh0 = int((x/img_degraded.shape[0]) * img.shape[0])
                xh1 = int(((x+w)/img_degraded.shape[0]) * img.shape[0])
                yh0 = int((y/img_degraded.shape[1]) * img.shape[1])
                yh1 = int(((y+h)/img_degraded.shape[1]) * img.shape[1])

                img_h = img[xh0:xh1, yh0:yh1, :]
                img_d = img_degraded[x:x+w, y:y+h, :]

                img_256 = self.resize_256(image=img_h)["image"]
                img_degraded = self.resize_256(image=img_d)["image"]
                crop = True
            else:
                crop = False
            img_resize = self.norm_torch(image=img_256)["image"]
            img_degraded = self.norm_torch(image=img_degraded)["image"]
            return img_resize, img_degraded, [cap], [name], crop

        # applied = np.random.choice(aug_list, size=n_iter, p=p, replace=False)
        # for name in applied:
        #     # print("--------------", _, name)
        #     # print("picked", name)
        #     t = self.map_aug[name]
        #     if name in ["Rain", "Fog"]:
        #         # img_256 = self.resize_256(image=img)["image"]
        #         if img_degraded is None:
        #             img_degraded = t(image=img_256)
        #         else:
        #             img_degraded = t(image=img_degraded)
        #         # img_degraded = self.norm_torch(image=img_d)["image"]
        #     else:
        #         if img_degraded is None:
        #             img_degraded = t(image=img_256)["image"]
        #         else:
        #             img_degraded = t(image=img_degraded)["image"]
                
        #     cap = self.random_generator.choice(len(self.cap_degraded[name]))
        #     cap = self.cap_degraded[name][cap]
        #     # print("cap", cap)
        #     # print("cur caps", caps)
        #     caps.append(cap)
        # if np.random.rand() < 0.3:
        #     # # crop 
        #     # w = np.random.randint(70, 200)
        #     # h = np.random.randint(70, 200)
        #     # x = np.random.randint(0, 255-w)
        #     # y = np.random.randint(0, 255-h)
        #     # img_256 = img_256[x:x+w, y:y+h, :]
        #     # img_degraded = img_degraded[x:x+w, y:y+h, :]
        #     # img_256 = self.resize_256(image=img_256)["image"]
        #     # img_degraded = self.resize_256(image=img_degraded)["image"]
        #     # crop = True
        #     crop = False
        # else:
        #     crop = False
        # img_resize = self.norm_torch(image=img_256)["image"]
        # img_degraded = self.norm_torch(image=img_degraded)["image"]
        # return img_resize, img_degraded, caps, applied, crop


class DenoiseDatasetText(Dataset):
    def __init__(self, mode="train", dataset_csv = [], num_patch=256, raw_latent = False, degrade_list=[], degrade_p = None, n_iter = 1):
        super(DenoiseDatasetText).__init__()
        self.boi = "<unused0>" # begin of image
        self.eoi = "<unused1>" # end of image
        self.mode = mode
        self.n_iter = n_iter
        self.num_patch = num_patch
        self.samples = [] # all samples, each sample has image_path and caption
        self.aug = AugmentDegraded()
        self.degrade_list = degrade_list
        if degrade_p is None:
            self.degrade_p = [1/len(degrade_list)] * len(degrade_list)
        else:
            self.degrade_p = degrade_p

        for csv_file in dataset_csv:
            raw = pd.read_csv(csv_file)
            # print(raw.head(5))
            # print(mode)
            all_samples = raw.values.tolist()
            # print(type(all_samples))
            # print(all_samples[:10])
            num_valid = int(min(1000, len(all_samples) * 0.20))
            if mode == "train":
                self.samples.extend(all_samples[:-num_valid])
                # self.samples = self.samples[:1000]
            elif mode == "valid" or mode == "validation":
                self.samples.extend(all_samples[-num_valid:])

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        pred_id = torch.arange(0, self.num_patch)
        img_path, cap_img = self.samples[index]
        # cap_img = self.captions[index]

        img = Image.open(img_path)
        img = img.convert("RGB")
        img = np.array(img)
        if isinstance(cap_img, str):
            crop_p = 0
        else:
            crop_p = 0.5
            cap_img = ""
        img_target, img_degraded, caps_degraded, applied, crop = self.aug(img=img, crop_p=crop_p, aug_list=self.degrade_list,p=self.degrade_p, n_iter=self.n_iter)
        
        # print("raw", caps_degraded)
        cap_degraded = ""
        for _ in caps_degraded:
            cap_degraded += _
            cap_degraded += " "
        # print("final", cap_degraded)
        if crop:
            cap_img = ""
        if img_degraded is not None:
            cap_final = cap_img + " " + cap_degraded + " " + "Enhance this image." + self.boi + self.eoi
            # cap_final = cap_img + " " + "Enhance this image." + self.boi + self.eoi
            # cap_final = cap_img + " " + cap_degraded + " " + "Output the mask." + self.boi + self.eoi
        else:
            cap_final = cap_img + " " + "Enhance this image." + self.boi + self.eoi
            img_degraded = img_target
            img_target = None
        # cap_final = "Enhance this image." + self.boi + self.eoi
        return {
            "pixel_values": img_degraded,
            "gt_images": img_target,
            "prompt": cap_final,
            "pred_patch_id": pred_id,
        }

class SegmentDatasetText(Dataset):
    def __init__(self, mode="train", dataset_csv = [], num_patch=256, raw_latent = False, degrade_list=[], degrade_p = None, n_iter=1, p_default_q = 0.1):
        super(SegmentDatasetText).__init__()
        self.boi = "<unused0>" # begin of image
        self.eoi = "<unused1>" # end of image
        self.mode = mode
        self.p_default_q = p_default_q
        self.n_iter = n_iter
        self.num_patch = num_patch
        self.samples = [] # all samples, each sample has image_path and caption
        self.aug = AugmentDegraded()
        self.degrade_list = degrade_list
        self.q = {
            "Blur": [
                "Segment region affected by blurriness",
                "Identify the noticeable blurring parts",
            ],
            "Compress": [
                "Segment region affected by compression artifacts",
                "Identify the blurry regions due to heavy compression",
            ],
            "Noise": [
                "Segment region affected by noise",
                "Identify the noisy regions",
            ],
            "Rain": [
                "Segment region affected by rain",
                "Identify the regions have noticeable degradation due to streaks resembling rain",
            ],
            "Fog": [
                "Segment region affected by haze or fog",
                "Identify the washed-out part, likely due to haze or fog",
            ]
        }
        if degrade_p is None:
            self.degrade_p = [1/len(degrade_list)] * len(degrade_list)
        else:
            self.degrade_p = degrade_p

        self.samples = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        # for csv_file in dataset_csv:
        #     raw = pd.read_csv(csv_file)
        #     all_samples = raw.values.tolist()
        #     num_valid = min(1000, len(all_samples) * 0.20)
        #     if mode == "train":
        #         self.samples.extend(all_samples[:-num_valid])
        #         # self.samples = self.samples[:1000]
        #     elif mode == "valid" or mode == "validation":
        #         self.samples.extend(all_samples[-num_valid:])

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        pred_id = torch.arange(0, self.num_patch)
        # img_path, cap_img = self.samples[index]
        # cap_img = self.captions[index]

        # img = Image.open(img_path)
        # cur_item = self.full_dataset[index]
        # img = cur_item["image"].convert("RGB")
        img = self.samples[index]["image"]
        img = img.convert("RGB")
        img = np.array(img)

        img_target, img_degraded, caps_degraded, applied = self.aug(img=img, aug_list=self.degrade_list,p=self.degrade_p, n_iter=self.n_iter)
        
        tmp = copy.deepcopy(img_target)
        mask = torch.zeros((256, 256))
        if applied is not None:
            for _ in range(2):
                w = np.random.randint(50, 220)
                h = np.random.randint(50, 220)
                x = np.random.randint(0, 255-w)
                y = np.random.randint(0, 255-h)

                tmp[:, x:x+w, y:y+h] = img_degraded[:, x:x+w, y:y+h]
                mask[x:x+w, y:y+h] = 1.0

            img_degraded = tmp
            pick = np.random.choice(applied)
            q_pick = np.random.choice(self.q[pick])
            if np.random.rand() < self.p_default_q:
                q_pick = np.random.choice([
                    "Segment the region needed to enhance",
                    "Identify the part of this image needed to enhance"
                ])
        else:
            pick = np.random.choice(list(self.q.keys()))
            q_pick = np.random.choice(self.q[pick])
            if np.random.rand() < self.p_default_q:
                q_pick = np.random.choice([
                    "Segment the region needed to enhance",
                    "Identify the part of this image needed to enhance"
                ])
        
        cap_final = q_pick + self.boi + self.eoi
        return {
            "pixel_values": img_degraded,
            # "gt_images": img_target,
            "prompt": cap_final,
            "pred_patch_id": pred_id,
            "mask": mask
        }

class CustomCollate():
    def __init__(self, cfg):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["model_path"])
        self.boi = cfg["llm"]["boi_token"] # begin of image
        self.eoi = cfg["llm"]["eoi_token"] # end of image
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [self.boi, self.eoi]
            },
            replace_additional_special_tokens=False
        ) # We leverage un-used tokens instead of adding new tokens, <unused0> = <img> <unused1> = </img>
    
    def __call__(self, batch):
        # print("log", batch.keys())
        ret = {}
        ret["pixel_values"] = torch.stack([p["pixel_values"] for p in batch], dim=0)
        ret["pred_patch_id"] = torch.stack([p["pred_patch_id"] for p in batch], dim=0)
        if "mask" in batch[0]:
            ret["mask"] = torch.stack([p["mask"] for p in batch], dim=0)
        if "gt_images" in batch[0]:
            if batch[0]["gt_images"] is not None:
                ret["gt_images"] = torch.stack([p["gt_images"] for p in batch], dim=0)
            else:
                ret["gt_images"] = None
        else:
            pass
        # for p in batch:
        #     print(p["prompt"])
        token = self.tokenizer([p["prompt"] for p in batch], padding=True, return_tensors="pt")
        ret["input_ids"] = token["input_ids"]
        ret["attention_masks"] = token["attention_mask"]
        return ret

class DenoiseDataset(Dataset):
    def __init__(self, mode="train", num_patch=256, raw_latent = False, degrade_transform=None):
        super(DenoiseDataset, self).__init__()
        self.mode = mode
        self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        self.crop = v2.Lambda(lambda pil_image: center_crop_arr(pil_image, 256))
        if degrade_transform is None:
            self.degrade_transform = v2.Compose([
                v2.RandomChoice([
                    v2.Compose([
                        v2.JPEG([10, 25]),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                    ]),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.RandomChoice([
                            # v2.GaussianBlur(kernel_size=5),
                            v2.GaussianBlur(kernel_size=7),
                            v2.GaussianBlur(kernel_size=9),
                        ]),
                    ]),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.RandomChoice([
                            v2.GaussianNoise(sigma=15.0/255.0),
                            v2.GaussianNoise(sigma=25.0/255.0),
                            v2.GaussianNoise(sigma=50.0/255.0),
                            # v2.GaussianNoise(sigma=0.2),
                            # v2.GaussianNoise(sigma=0.25),
                            # v2.GaussianNoise(sigma=0.3),
                            # v2.GaussianNoise(sigma=0.35),
                            # v2.GaussianNoise(sigma=0.4),
                            # v2.GaussianNoise(sigma=0.45),
                            # v2.GaussianNoise(sigma=0.5),
                        ]),
                    ]),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                    ])
                ], p=[1.0, 0.0, 0.0, 0.0]),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.degrade_transform=degrade_transform
        # self.Norm
        self.norm = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.patch_size = 1
        self.num_patch = num_patch
        self.raw_latent = raw_latent

    def __len__(self):
        return len(self.full_dataset)
    
    def patchify(self, x):
        # x = torch.unsqueeze(x, 0)
        # print("before patch", x.shape)
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        # print("after patch", x.shape)
        x = x[0]
        return x  # [l, d]

    def __getitem__(self, index):
        # pred_id = torch.randint(low=0, high=256, size=[1])
        pred_id = torch.arange(0, self.num_patch)
        # print("log", pred_id.shape)
        # load gt latent
        # file = open(f"/scratch2/f0072r1/img_latent/{self.mode}/latent_{index}.pt",'rb')
        if self.raw_latent:
            data = np.load(f"/scratch2/f0072r1/kl-16-latent/{self.mode}/{index}.npz")
            moments = torch.from_numpy(data["moments"]).unsqueeze(0).to(torch.float32)
            posterior = DiagonalGaussianDistribution(moments)
            gt_latent = posterior.sample().mul_(0.2325)
            gt_latent = self.patchify(gt_latent)
        else:
            gt_latent = None
        # gt_latent = pickle.load(file)
        # file.close()
        # gt_latent = gt_latent[pred_id[0]]
        
        
        # print("88888888888888888888", gt_latent.shape)
        
        # print("88888888888888888888", gt_latent.shape)
        # gt_latent = gt_latent[pred_id[0]]
        # gt_latent = torch.from_numpy(gt_latent).to(torch.float32)
        cur_item = self.full_dataset[index]
        img = cur_item["image"].convert("RGB")
        img_256 = self.crop(img)
        img_noise = self.degrade_transform(img_256)
        img_256_norm = self.norm(img_256)
        # img_noise = self.noise(img)
        # img_noise = self.norm(img_noise)
        
        # img_clean = self.norm(img)
        if self.raw_latent:
            return {"pixel_values":img_noise, "pred_patch_id": pred_id, "gt_latent": gt_latent, "gt_images": img_256_norm}
        else:
            return {"pixel_values":img_noise, "pred_patch_id": pred_id, "gt_images": img_256_norm}

class SpecificDataset(Dataset):
    def __init__(self, dataname = "CBSD68", num_patch=256, transform=None):
        super(SpecificDataset, self).__init__()
        # self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        self.full_dataset = []
        self.num_patch = num_patch
        if dataname == "CBSD68":
            for filename in os.listdir("/scratch2/f0072r1/img_dataset/CBSD68-dataset/CBSD68/noisy50"):
                full_path = os.path.join("/scratch2/f0072r1/img_dataset/CBSD68-dataset/CBSD68/original_png", filename)
                full_path_noise = os.path.join("/scratch2/f0072r1/img_dataset/CBSD68-dataset/CBSD68/noisy50", filename)
                img_clean = Image.open(full_path)
                img_noise = Image.open(full_path_noise)
                self.full_dataset.append({
                    "image": img_clean,
                    "image_noise": img_noise
                })

        # self.transform = v2.Compose([
        #     v2.ToImage(), 
        #     v2.Resize((256, 256)),
        #     # v2.RGB(),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.GaussianNoise(sigma=50.0/255.0),
        #     v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])
        self.transform = transform

        self.norm = v2.Compose([
            v2.ToImage(),
            v2.Resize((256, 256)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    
    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, index):
        # pred_id = torch.randint(low=0, high=256, size=[1])
        pred_id = torch.arange(0, self.num_patch)
        cur_item = self.full_dataset[index]
        # img = cur_item["image_noise"].convert("RGB")
        # img = self.transform(img)
        
        # img_clean = img
        # img_noise = self.noise(img)
        # img_noise = cur_item["image_noise"].convert("RGB")
        img_clean = cur_item["image"].convert("RGB")
        img_noise = self.transform(img_clean)
        img_clean = self.norm(img_clean)
        # img_noise = self.norm(img_noise)
        # img_noise = self.norm(img_noise)
        # img_clean = self.norm(img)
        return {"pixel_values":img_noise, "gt_images":img_clean, "pred_patch_id": pred_id}

class CustomCollateMulti():
    def __init__(self, cfg):
        # print(cfg["llm"]["model_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["llm"]["model_path"])
        self.boi = cfg["llm"]["boi_token"] # begin of image
        self.eoi = cfg["llm"]["eoi_token"] # end of image
        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [self.boi, self.eoi]
            },
            replace_additional_special_tokens=False
        ) # We leverage un-used tokens instead of adding new tokens, <unused0> = <img> <unused1> = </img>
    
    def __call__(self, batch):
        ret = {}
        ret["pixel_values"] = torch.stack([p["pixel_values"] for p in batch], dim=0)
        ret["target_image"] = torch.stack([p["target_image"] for p in batch], dim=0)
        token_t2i = self.tokenizer([p["prompts"]["question_t2i"] for p in batch], padding=True, return_tensors="pt")
        ret["input_ids_t2i"] = token_t2i["input_ids"]
        ret["attention_masks_t2i"] = token_t2i["attention_mask"]
        # token_cap = 
        temp_p = []
        temp_l = []
        temp_p_wo = []
        for question, answer in zip([p["prompts"]["question_i2t"] for p in batch], [p["prompts"]["answer"] for p in batch]):
            full_sentence = question +" "+ answer + "<end_of_turn>"
            temp_p_wo.append(question +" ")
            full_sentence_pt = self.tokenizer(full_sentence, return_tensors="pt")
            question_len = self.tokenizer(question, return_tensors="pt").input_ids.ne(self.tokenizer.pad_token_id).sum().item()
            # print("full sentence", full_sentence)
            # print(self.tokenizer.decode(full_sentence[0], skip_special_tokens=False))
            # print(self.tokenizer.decode(full_sentence[0][:question_len], skip_special_tokens=False))
            # print("q len", question_len)
            input_ids = full_sentence_pt["input_ids"].flatten()
            # source_len = question["input_ids_lens"]
            labels = copy.deepcopy(input_ids)
            labels[ :question_len] = -100
            temp_p.append(full_sentence)
            temp_l.append(labels)
        tmp = self.tokenizer(temp_p, padding=True, return_tensors="pt")
        # print(tmp)
        tmp1 = self.tokenizer(temp_p_wo, padding=True, return_tensors="pt")
        ret["input_ids_i2t_q"] = tmp1["input_ids"]
        ret["attention_masks_i2t_q"] = tmp1["attention_mask"]
        ret["input_ids_i2t"] = tmp["input_ids"]
        ret["attention_masks_i2t"] = tmp["attention_mask"]
        tmp_label = []
        for p, l in zip(ret["input_ids_i2t"], temp_l):
            if p.shape[0] != l.shape[0]:
                l = torch.concat([torch.from_numpy(np.array([-100] * (p.shape[0] - l.shape[0]))).to(l.dtype).to(l.device), l])
            # l = torch.concat([torch.from_numpy(np.array([-100] * 1024)).to(l.dtype).to(l.device), l])
            tmp_label.append(l)
        tmp_label = torch.stack(tmp_label)
        ret["labels"] = tmp_label
            # print(p)
            # print(l)
            # print(l.shape, p.shape)
            # print()
            # print(input_ids)
            # print(labels)
        # token_task0 = self.tokenizer([p["prompts"][0] for p in batch], padding=True, return_tensors="pt")
        # ret["input_ids0"] = token_task0["input_ids"]
        # ret["attention_masks0"] = token_task0["attention_mask"]
        return ret

class ImageCaptioning(Dataset):
    def __init__(self, mode, dataset_dict, degrade_list, degrade_p, n_iter, stage1 = False,):
        super(ImageCaptioning).__init__()
        self.boi = "<unused0>" # begin of image
        self.eoi = "<unused1>" # end of image
        self.stage1 = stage1
        random.seed(1234)
        with open(dataset_dict, "r") as f:
            self.ds = json.load(f)
        random.shuffle(self.ds)
        if mode != "train":
            self.ds = self.ds[-5000:]
        self.length = len(self.ds)
        self.aug = AugmentDegraded()
        self.degrade_list = degrade_list
        self.degrade_p = degrade_p
        self.n_iter = n_iter
        self.resize_norm_torch = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.ToTensorV2(),
        ])
    def __len__(self,):
        return self.length
    
    def __getitem__(self, index):
        image = self.ds[index]["hq_path"]
        cap_img = self.ds[index]["caption"]
        
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        image = np.array(image)
        # print(self.ds[index]["lq"])
        if self.ds[index]["lq"] != {}:
            # print("32094290342374987239847298374987234987238974982734897239847298374")
            image_lq = Image.open(self.ds[index]["lq"]["lq_path"])
            name = self.ds[index]["lq"]["type"]
            image_lq = image_lq.convert("RGB")
            image_lq = np.array(image_lq)
            
            caps_degraded = [self.aug.random_generator.choice(self.aug.cap_degraded[name])]
            image_hq = self.resize_norm_torch(image=image)["image"]
            image_lq = self.resize_norm_torch(image=image_lq)["image"]
            crop = False
        else:    
            if isinstance(cap_img, str) and len(cap_img) > 0:
                crop_p = 0
            else:
                crop_p = 0.4
                cap_img = ""
            image_hq, image_lq, caps_degraded, applied, crop = self.aug(img=image, crop_p=crop_p, aug_list=self.degrade_list,p=self.degrade_p, n_iter=self.n_iter)
        
        # print("raw", caps_degraded)
        cap_degraded = ""
        for _ in caps_degraded:
            cap_degraded += _
            cap_degraded += " "
        # print("final", cap_degraded)
        if crop:
            cap_img = ""
        
        cap_final = cap_img + " " + cap_degraded + " " + "Enhance this image!"
            
        prompt = {
            # "question_t2i": self.boi + self.eoi + "Describe the image and output the same image." + self.boi, 
            # "question_i2t": self.boi + self.eoi + "Describe the image and output the same image.", 
            "question_t2i": self.boi + self.eoi + cap_final + self.boi, 
            "question_i2t": self.boi + self.eoi + "Assess the quality of this image?", 
            # "answer": cap + ' ' + cap_d
            "answer": cap_degraded
        }
            
        return {
            "pixel_values": image_lq,
            "prompts": prompt,
            "target_image": image_hq,
        }

class TestReal(Dataset):
    def __init__(self, mode, dataset_dict, add_noise = False):
        super(TestReal).__init__()
        self.boi = "<unused0>" # begin of image
        self.eoi = "<unused1>" # end of image
        random.seed(1234)
        self.add_noise = add_noise
        with open(dataset_dict, "r") as f:
            self.ds = json.load(f)
        self.length = len(self.ds)
        self.aug = AugmentDegraded()
        #self.degrade_list = degrade_list
        # self.degrade_p = degrade_p
        # self.n_iter = n_iter
        self.resize_norm_torch = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.ToTensorV2(),
        ])
        self.resize_noise_norm_torch = A.Compose([
            A.Resize(256, 256),
            A.GaussNoise(std_range=(30/255.0, 30/255.0), p=1.0),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.ToTensorV2(),
        ])
        
    def __len__(self,):
        return self.length
    
    def __getitem__(self, index):
        image_hq = self.ds[index]["hq_path"]
        cap_img = self.ds[index]["caption"]
        # image_lq = self.ds[index]["lq"]["lq_path"]
        # print("log hq path", image_hq)
        if isinstance(image_hq, str) and len(image_hq) > 0:
            # image_lq = Image.open(image_lq)
            image_hq = Image.open(image_hq)
            image_hq = image_hq.convert("RGB")
            image_hq = np.array(image_hq)
        else:
            image_hq = None
        if self.add_noise:
            image_lq = self.resize_noise_norm_torch(image=image_hq)["image"]
            name = "noise"
        else:
            image_lq = self.ds[index]["lq"]["lq_path"]
            image_lq = Image.open(image_lq)
            image_lq = image_lq.convert("RGB")
            image_lq = np.array(image_lq)
            image_lq = self.resize_norm_torch(image=image_lq)["image"]
            name = self.ds[index]["lq"]["type"]
        
        # print(self.ds[index]["lq"])
        
            # print("32094290342374987239847298374987234987238974982734897239847298374")
        
            
        caps_degraded = self.aug.random_generator.choice(self.aug.cap_degraded[name])
        # image_hq = self.resize_norm_torch(image=image)["image"]
        
        if image_hq is not None:
            image_hq = self.resize_norm_torch(image=image_hq)["image"]
        else:
            image_hq = image_lq
        
        cap_final = cap_img + " " + caps_degraded + " " + "Enhance this image!"
            
        prompt = {
            # "question_t2i": self.boi + self.eoi + "Describe the image and output the same image." + self.boi, 
            # "question_i2t": self.boi + self.eoi + "Describe the image and output the same image.", 
            "question_t2i": self.boi + self.eoi + cap_final + self.boi, 
            "question_i2t": self.boi + self.eoi + "Assess the quality of this image?", 
            # "answer": cap + ' ' + cap_d
            "answer": ""
        }
            
        return {
            "pixel_values": image_lq,
            "prompts": prompt,
            "target_image": image_hq,
        }

class CustomDataset(Dataset):
    def __init__(self, files = None, num_patch=256, transform=None):
        super(CustomDataset, self).__init__()
        # self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        self.full_dataset = []
        self.num_patch = num_patch
        for img_path in files:        
            img_clean = Image.open(img_path)
            self.full_dataset.append({
                "image": img_clean,
            })

        self.transform = transform
    
    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, index):
        pred_id = torch.arange(0, self.num_patch)
        cur_item = self.full_dataset[index]
        img = cur_item["image"].convert("RGB")
        img = self.transform(img)
        return {"pixel_values":img, "pred_patch_id": pred_id}

class RawDataset(Dataset):
    def __init__(self, mode="train"):
        super(RawDataset, self).__init__()
        self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        self.transform = v2.Compose([
            v2.ToImage(), 
            v2.Resize((224, 224)),
            # v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.norm = v2.Normalize(mean=IMG_MEAN, std=IMG_STD)

    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, index):
        cur_item = self.full_dataset[index]
        img = cur_item["image"].convert("RGB")
        img = self.transform(img)
        img_clean = self.norm(img)
        return {"pixel_values": img_clean}

class TestZoomIn(Dataset):
    def __init__(self, mode, degrade_list, degrade_p, n_iter, stage1 = False,zoom_p=1.0):
        super(ImageCaptioning).__init__()
        self.boi = "<unused0>" # begin of image
        self.eoi = "<unused1>" # end of image
        self.stage1 = stage1
        self.zoom_p = zoom_p
        random.seed(1234)
        self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        # random.shuffle(self.ds)
        # if mode != "train":
        #     self.ds = self.ds[-5000:]
        # self.length = len(self.ds)
        self.aug = AugmentDegraded()
        self.degrade_list = degrade_list
        self.degrade_p = degrade_p
        self.n_iter = n_iter
        self.resize_norm_torch = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            A.ToTensorV2(),
        ])

    def __len__(self,):
        return len(self.full_dataset)
    
    def __getitem__(self, index):
        cur_item = self.full_dataset[index]
        img = cur_item["image"].convert("RGB")
        img = np.array(img)
        crop_p = self.zoom_p
        image_hq, image_lq, caps_degraded, applied, crop = self.aug(img=img, crop_p=crop_p, aug_list=self.degrade_list,p=self.degrade_p, n_iter=self.n_iter)
        cap_degraded = ""
        for _ in caps_degraded:
            cap_degraded += _
            cap_degraded += " "
        # print("final", cap_degraded)
        if crop:
            cap_img = ""
        else:
            cap_img = ""
        
        cap_final = cap_img + " " + cap_degraded + " " + "Enhance this image!"
            
        prompt = {
            # "question_t2i": self.boi + self.eoi + "Describe the image and output the same image." + self.boi, 
            # "question_i2t": self.boi + self.eoi + "Describe the image and output the same image.", 
            "question_t2i": self.boi + self.eoi + cap_final + self.boi, 
            "question_i2t": self.boi + self.eoi + "Assess the quality of this image?", 
            # "answer": cap + ' ' + cap_d
            "answer": cap_degraded
        }
            
        return {
            "pixel_values": image_lq,
            "prompts": prompt,
            "target_image": image_hq,
        }

if __name__ == "__main__":
    test = DenoiseDataset(mode = "train")
    print("23987498237498723984798237489237498723984798237489273984723987")
    for step, item in enumerate(range(10)):
        # print(item)
        # print(type(item["pixel_values"]), item["pixel_values"].shape)
        sample = test.__getitem__(item)
        print(type(sample["pixel_values"]), sample["pixel_values"].shape)
        save_image(sample["pixel_values"], f"sample_{step}.png", normalize=True, value_range=(-1, 1))
        if step > 3:
            break
    
    # test_dataloader = torch.utils.data.DataLoader(
    #     test,
    #     batch_size=4,
    #     num_workers=8,
    #     shuffle=True,
    # )

    # for step, batch in enumerate(test_dataloader):
    #     print(batch['gt_latent'].shape)
    #     print(batch['pixel_values'].shape)
    #     print(batch["pred_patch_id"].shape)
    #     cur = item
    #     break
    # cur = cur["img_noise"]
    # print(cur.shape)
    # cur = torch.unsqueeze(cur, 0)
    # cur = denormalize(cur)
    # show_image(cur[0])
    # plt.savefig("myImagePDF.pdf", format="pdf", bbox_inches="tight")
