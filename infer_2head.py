import os 
import sys
sys.path.insert(0, os.path.abspath('.'))
import torch
import wandb
import requests
import logging
import pickle
import math
import time

import numpy as np

from torchvision.utils import save_image, make_grid
from src.model import ResLLM, ResLLMDiff, ResLLMAE, ResLLMAEText, ResLLM2Heads
from src.custom_data import RawDataset, DenoiseDataset, DenoiseDatasetText, CustomCollate, ImageCaptioning, CustomCollateMulti, TestReal, TestZoomIn
from src.utils import parse_args, norm_range
from src.metrics import calculate_psnr_pt, calculate_ssim_pt
from datasets import load_dataset
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import tqdm
from transformers import get_scheduler, AutoImageProcessor
from PIL import Image
from accelerate import DistributedDataParallelKwargs



logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = get_logger(__name__, log_level="INFO")

def eval_model(model, eval_dataloader, current_iter, accelerator, logger, tokenizer):
    start_time = time.time()
    model.eval()
    is_print = False
    samples_seen = 0
    all_psnr = []
    all_ssim = []
    all_samples = []
    all_prompt = []
    all_answer = []
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            progress_bar.update(1)
            
            # gen_text = model(
            #     input_ids_t2i = batch["input_ids_i2t_q"],
            #     attention_masks_t2i = batch["attention_masks_i2t_q"],
            #     pixel_values= batch["pixel_values"],
            #     gen_text_only = True
            # )
            # gen_text = tokenizer.batch_decode(gen_text, skip_special_tokens=False)
            # all_answer.extend(gen_text)
            gen_image = model(
                input_ids_t2i = batch["input_ids_t2i"],
                attention_masks_t2i = batch["attention_masks_t2i"],
                pixel_values= batch["pixel_values"],
                gen_image_only = True
            )
            pred_img = gen_image
            input_imgs = batch["pixel_values"]
            # prompt = batch["input_ids_t2i"]
            prompts = tokenizer.batch_decode(batch["input_ids_t2i"], skip_special_tokens=False)
            all_prompt.extend(prompts)
            
            gt_img = batch["target_image"]
            input_imgs = norm_range(input_imgs, (-1, 1))
            pred_unnorm = norm_range(pred_img, (-1, 1))
            gt_unnorm = norm_range(gt_img, (-1,1))
            img_save = torch.concat([input_imgs, pred_unnorm], dim=3)
            for id in range(img_save.shape[0]):
                all_samples.append(make_grid(img_save[id], nrow=1, normalize=False))
            psnr = calculate_psnr_pt(pred_unnorm, gt_unnorm, crop_border=4)
            ssim = calculate_ssim_pt(pred_unnorm, gt_unnorm, crop_border=4)
            # total_loss += loss.detach().float()
            # loss_d = accelerator.gather((loss_d))
            # loss_g = accelerator.gather((loss_g))
            psnr = accelerator.gather((psnr))
            ssim = accelerator.gather((ssim))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    # loss_g = loss_g[: len(eval_dataloader.dataset) - samples_seen]
                    # loss_d = loss_d[: len(eval_dataloader.dataset) - samples_seen]
                    psnr = psnr[: len(eval_dataloader.dataset) - samples_seen]
                    ssim = ssim[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += batch["pixel_values"].shape[0]
            # all_loss_g.append(loss_g)
            # all_loss_d.append(loss_d)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
        # logger.info(f"{loss}, {loss.dtype}")
    # all_loss_g = torch.cat(all_loss_g, 0)
    # all_loss_g = torch.mean(all_loss_g)
    # all_loss_d = torch.cat(all_loss_d, 0)
    # all_loss_d = torch.mean(all_loss_d)
    all_psnr = torch.cat(all_psnr, 0)
    # print("------------------- all_psnr", all_psnr)
    logger.info(f"PSNR: {all_psnr}")
    all_psnr = torch.mean(all_psnr)
    all_ssim = torch.cat(all_ssim, 0)
    # print("-------------------- all_ssim", all_ssim)
    all_ssim = torch.mean(all_ssim)
    tmp = np.random.choice(len(all_samples), size=10)
    # print(tmp)
    all_samples = [all_samples[p] for p in tmp]
    all_prompt = [all_prompt[p] for p in tmp]
    # all_answer = [all_answer[p] for p in tmp]
    all_answer = [""] * len(all_prompt)
    logger.info(f"PSNR: {all_psnr} SSIM: {all_ssim}")
    model.train()
    # return all_loss_g, all_loss_d, all_psnr, all_ssim, all_samples, all_prompt
    # return all_psnr, all_ssim, all_samples, all_prompt, all_answer
    return all_psnr, all_ssim, all_samples, all_prompt, all_answer

def main():
    args, cfg = parse_args()
    # print(args)
    if args.with_tracking:
        # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")
        accelerator = Accelerator(log_with="wandb")
        logger.info(f"enable Wandb", main_process_only=True)
        if accelerator.is_main_process:
            experiment_config = cfg
            accelerator.init_trackers(cfg["project"], experiment_config)
    else:
        # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        accelerator = Accelerator()
        logger.info(f"No Wandb", main_process_only=True)
    
    logger.info(f"{cfg}")
    logger.info(f"{accelerator.state}")
    
    collate_fn = CustomCollateMulti(cfg)
    with accelerator.main_process_first():
        # train_ds = ImageCaptioning(mode="train", stage1=True)
        # dev_ds = TestReal(
        #     mode="validation",
        #     # dataset_dict="/scratch2/f0072r1/res_gemma/full_dataset_test.json", 
        #     # dataset_dict="/scratch2/f0072r1/res_gemma/full_dataset_test_0.json", 
        #     dataset_dict="/scratch2/f0072r1/res_gemma/full_haze_indoor.json",
            
        #     add_noise=False
        # )
        # # ImageCaptioning(mode="validation", stage1=True)
        # dev_dataloader = torch.utils.data.DataLoader(
        #     dataset=dev_ds,
        #     batch_size=16,
        #     num_workers=8,
        #     collate_fn=collate_fn,
        #     shuffle=False
        # )

        dev_ds = TestZoomIn(
            mode="validation",
            # dataset_dict="/scratch2/f0072r1/res_gemma/full_data.json", 
            degrade_list=["noise", "blur", "compress", "rain", "fog", "sr" ],
            degrade_p=[0.20,       0.10,     0.20,       0.20,  0.20,  0.10],
            n_iter=1,
            zoom_p=0.0
        )
        # ImageCaptioning(mode="validation", stage1=True)
        dev_dataloader = torch.utils.data.DataLoader(
            dataset=dev_ds,
            batch_size=cfg["optimizer"]["per_device_dev_batch_size"],
            num_workers=cfg["optimizer"]["per_device_workers"],
            collate_fn=collate_fn,
            shuffle=False
        )
    
    model = ResLLM2Heads(cfgs=cfg)
    # model = ResLLMDiff(cfgs=cfg)
    # model.init(cfg, ckpt_dir="/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-it-alltype-text-lora32-parallel-pix-percept-latent/step_1_864", is_trainable=True)
    # model.init(cfg)
    # model.init(cfg, ckpt_dir="/scratch2/f0072r1/res_gemma/logs/gemma2-2b-lora32-stage-2-imageonly/step_15_3200")
    model.init(cfg, ckpt_dir="/scratch2/f0072r1/res_gemma/logs/gemma2-2b-lora32-stage-2-imageonly-1/step_2_12000")
    
    model, dev_dataloader = accelerator.prepare(
        model, dev_dataloader
    )

    tokenizer = collate_fn.tokenizer
    model.eval()
    is_print = False
    samples_seen = 0
    all_psnr = []
    all_ssim = []
    all_samples = []
    all_prompt = []
    all_answer = []
    progress_bar = tqdm(range(len(dev_dataloader)), disable=not accelerator.is_local_main_process)
    os.makedirs("/scratch2/f0072r1/res_gemma/output_imgs/real2", exist_ok=True)
    with torch.no_grad():
        for step, batch in enumerate(dev_dataloader):
            progress_bar.update(1)
            
            # gen_text = model(
            #     input_ids_t2i = batch["input_ids_i2t_q"],
            #     attention_masks_t2i = batch["attention_masks_i2t_q"],
            #     pixel_values= batch["pixel_values"],
            #     gen_text_only = True
            # )
            # gen_text = tokenizer.batch_decode(gen_text, skip_special_tokens=False)
            # all_answer.extend(gen_text)
            gen_image = model(
                input_ids_t2i = batch["input_ids_t2i"],
                attention_masks_t2i = batch["attention_masks_t2i"],
                pixel_values= batch["pixel_values"],
                gen_image_only = True
            )
            pred_img = gen_image
            input_imgs = batch["pixel_values"]
            # prompt = batch["input_ids_t2i"]
            prompts = tokenizer.batch_decode(batch["input_ids_t2i"], skip_special_tokens=False)
            all_prompt.extend(prompts)
            
            gt_img = batch["target_image"]
            input_imgs = norm_range(input_imgs, (-1, 1))
            pred_unnorm = norm_range(pred_img, (-1, 1))
            gt_unnorm = norm_range(gt_img, (-1,1))
            img_save = torch.concat([input_imgs, pred_unnorm], dim=3)
            # img_save = pred_unnorm
            for id in range(img_save.shape[0]):
                all_samples.append(make_grid(img_save[id], nrow=1, normalize=False))
                save_image(img_save[id], f"/scratch2/f0072r1/res_gemma/output_imgs/real2/img_pred_{step}_{id}.png", normalize=False)
                save_image(input_imgs[id], f"/scratch2/f0072r1/res_gemma/output_imgs/real2/img_inp_{step}_{id}.png", normalize=False)
            psnr = calculate_psnr_pt(pred_unnorm, gt_unnorm, crop_border=4)
            ssim = calculate_ssim_pt(pred_unnorm, gt_unnorm, crop_border=4)
            # total_loss += loss.detach().float()
            # loss_d = accelerator.gather((loss_d))
            # loss_g = accelerator.gather((loss_g))
            psnr = accelerator.gather((psnr))
            ssim = accelerator.gather((ssim))
            if accelerator.num_processes > 1:
                if step == len(dev_dataloader) - 1:
                    # loss_g = loss_g[: len(eval_dataloader.dataset) - samples_seen]
                    # loss_d = loss_d[: len(eval_dataloader.dataset) - samples_seen]
                    psnr = psnr[: len(dev_dataloader.dataset) - samples_seen]
                    ssim = ssim[: len(dev_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += batch["pixel_values"].shape[0]
            # all_loss_g.append(loss_g)
            # all_loss_d.append(loss_d)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
        # logger.info(f"{loss}, {loss.dtype}")
    # all_loss_g = torch.cat(all_loss_g, 0)
    # all_loss_g = torch.mean(all_loss_g)
    # all_loss_d = torch.cat(all_loss_d, 0)
    # all_loss_d = torch.mean(all_loss_d)
    all_psnr = torch.cat(all_psnr, 0)
    print("------------", all_psnr)
    all_psnr = torch.mean(all_psnr)
    all_ssim = torch.cat(all_ssim, 0)
    all_ssim = torch.mean(all_ssim)
    # for item in all_samples:
        
    tmp = np.random.choice(len(all_samples), size=10)
    # # print(tmp)
    # all_samples = [all_samples[p] for p in tmp]
    all_prompt = [all_prompt[p] for p in tmp]
    # all_answer = [all_answer[p] for p in tmp]
    all_answer = [""] * len(all_prompt)
    logger.info(f"PSNR: {all_psnr} SSIM: {all_ssim}")

if __name__ == "__main__":
    main()

    






# for epoch in range(0, cfg["optimizer"]["num_train_epochs"]):
    #     model.train()
    #     for step, batch in enumerate(train_dataloader):
    #         batch["current_iter"] = completed_steps
    #         outputs = model(**batch)
    #         loss = outputs["loss"]
    #         print(loss)
    #         if "gan_opt" in cfg["loss"]:
    #             for p in model.module.net_d.parameters():
    #                 p.requires_grad = False
    #         loss_g = loss["recon_loss"]
    #         loss_g = loss_g / cfg["optimizer"]["gradient_accumulation_steps"]
    #         accelerator.backward(loss_g)
    #         if "gan_opt" in cfg["loss"]:
    #             for p in model.module.net_d.parameters():
    #                 p.requires_grad = True
    #             if "disc_loss" in loss:
    #                 loss_d = loss["disc_loss"]
    #                 loss_d /= cfg["optimizer"]["gradient_accumulation_steps"]
    #                 accelerator.backward(loss_d)
    #             else:
    #                 loss_d = None
            

    #         if (step + 1) % cfg["optimizer"]["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
    #             completed_steps += 1
    #             optimizer.step()
    #             if "gan_opt" in cfg["loss"]:
    #                 optimizer_d.step()
    #             optimizer.zero_grad()
    #             if "gan_opt" in cfg["loss"]:
    #                 optimizer_d.zero_grad()
    #         if step > 20:
    #             break
    #     break
