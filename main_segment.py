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
from src.model import ResLLM, ResLLMDiff, ResLLMAE, ResLLMAEText, ResTextSegment
from src.custom_data import RawDataset, DenoiseDataset, DenoiseDatasetText, CustomCollate, SegmentDatasetText
from src.utils import parse_args, norm_range
from src.metrics import calculate_psnr_pt, calculate_ssim_pt
from datasets import load_dataset
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import tqdm
from transformers import get_scheduler, AutoImageProcessor
from PIL import Image

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = get_logger(__name__, log_level="INFO")

def eval_model(model, eval_dataloader, accelerator, logger, tokenizer):
    start_time = time.time()
    model.eval()
    is_print = False
    samples_seen = 0
    all_loss = []
    # all_psnr = []
    # all_ssim = []
    all_samples = []
    all_prompt = []
    progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            progress_bar.update(1)
            outputs = model(**batch)
            loss = outputs["loss"]
            input_imgs = batch["pixel_values"]
            
            prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            all_prompt.extend(prompts)
            pred_img = outputs["pred_img"]
            mask = batch["mask"]
            mask = torch.unsqueeze(mask, 1).expand(-1, 3, -1, -1)
            input_imgs = norm_range(input_imgs, (-1, 1))
            # pred_unnorm = norm_range(pred_img, (-1, 1))
            pred_unnorm = pred_img
            # gt_unnorm = norm_range(gt_img, (-1,1))
            img_save = torch.concat([input_imgs, pred_unnorm, mask], dim=3)
            for id in range(img_save.shape[0]):
                all_samples.append(make_grid(img_save[id], nrow=1, normalize=False))
            # psnr = calculate_psnr_pt(pred_unnorm, gt_unnorm, crop_border=4)
            # ssim = calculate_ssim_pt(pred_unnorm, gt_unnorm, crop_border=4)
            # total_loss += loss.detach().float()
            loss = accelerator.gather((loss))
            # psnr = accelerator.gather((psnr))
            # ssim = accelerator.gather((ssim))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    loss = loss[: len(eval_dataloader.dataset) - samples_seen]
                    # psnr = psnr[: len(eval_dataloader.dataset) - samples_seen]
                    # ssim = ssim[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += loss.shape[0]
            all_loss.append(loss)
            # all_psnr.append(psnr)
            # all_ssim.append(ssim)
        # logger.info(f"{loss}, {loss.dtype}")
    all_loss = torch.cat(all_loss, 0)
    all_loss = torch.mean(all_loss)
    # all_psnr = torch.cat(all_psnr, 0)
    # all_psnr = torch.mean(all_psnr)
    # all_ssim = torch.cat(all_ssim, 0)
    # all_ssim = torch.mean(all_ssim)
    tmp = np.random.default_rng(1234).choice(len(all_samples), size=10)
    # print(tmp)
    all_samples = [all_samples[p] for p in tmp]
    all_prompt = [all_prompt[p] for p in tmp]
    # logger.info(f"{all_loss}")
    return all_loss, all_samples, all_prompt

def main():
    args, cfg = parse_args()
    # print(args)

    if args.with_tracking:
        accelerator = Accelerator(log_with="wandb")
        logger.info(f"enable Wandb", main_process_only=True)
        if accelerator.is_main_process:
            experiment_config = cfg
            accelerator.init_trackers(cfg["project"], experiment_config)
    else:
        accelerator = Accelerator()
        logger.info(f"No Wandb", main_process_only=True)
    logger.info(f"{cfg}")
    logger.info(f"{accelerator.state}")
    
    # train_dataset = DenoiseDatasetText(
    #     mode="train",
    #     dataset_csv=["/scratch2/f0072r1/res_gemma/pixmo_150_filter.csv"],
    #     degrade_list=["Blur", "Compress", "Noise", "Rain", "SR", "Fog"],
    #     num_patch= 32 * 32
    # )
    train_dataset = SegmentDatasetText(
        mode="train",
        dataset_csv=["/scratch2/f0072r1/res_gemma/pixmo_filter.csv"],
        degrade_list=["Blur", "Compress", "Noise", "Rain", "Fog"],
        num_patch= 32 * 32,
        n_iter=2,
        p_default_q=0.2
    )
    custom_collate = CustomCollate(cfg=cfg)
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        collate_fn=custom_collate,
        batch_size=cfg["optimizer"]["per_device_train_batch_size"],
        num_workers=cfg["optimizer"]["per_device_workers"],
        shuffle=True,
    )
    dev_dataset = SegmentDatasetText(
        mode="validation",
        dataset_csv=["/scratch2/f0072r1/res_gemma/pixmo_filter.csv"],
        degrade_list=["Blur", "Compress", "Noise", "Rain", "Fog"],
        num_patch= 32 * 32,
        n_iter=2,
        p_default_q=0.2
    )
    # dev_dataset = DenoiseDatasetText(
    #     mode="valid",
    #     dataset_csv=["/scratch2/f0072r1/res_gemma/pixmo_150_filter.csv"],
    #     degrade_list=["Blur", "Compress", "Noise", "Rain", "SR", "Fog"],
    #     num_patch= 32 * 32
    # )
    dev_dataloader = torch.utils.data.DataLoader(
        dataset=dev_dataset,
        collate_fn=custom_collate,
        batch_size=cfg["optimizer"]["per_device_dev_batch_size"],
        num_workers=cfg["optimizer"]["per_device_workers"],
        shuffle=False,
    )
    
    
    # model = ResLLMAEText(cfgs=cfg)
    model = ResTextSegment(cfgs=cfg)
    # model = ResLLMDiff(cfgs=cfg)
    # model.init(cfg, ckpt_dir="/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-it-alltype-text-lora32-parallel-pix-percept-latent/step_1_864", is_trainable=True)
    # model.init(cfg)
    model.init(cfg, ckpt_dir="/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-lora32-dice-4/step_0_1000", is_trainable=True)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["optimizer"]["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    total_batch_size = cfg["optimizer"]["per_device_train_batch_size"] * accelerator.num_processes * cfg["optimizer"]["gradient_accumulation_steps"]
    cfg["optimizer"]["learning_rate"] = float(cfg["optimizer"]["based_learning_rate"]) * total_batch_size / 256
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg["optimizer"]["learning_rate"], betas=(0.9, 0.95))

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg["optimizer"]["gradient_accumulation_steps"])
    if args.max_train_steps is None:
        args.max_train_steps = cfg["optimizer"]["num_train_epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=cfg["optimizer"]["lr_scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=cfg["optimizer"]["num_warmup_steps"],
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg["optimizer"]["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        args.max_train_steps = cfg["optimizer"]["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg["optimizer"]["num_train_epochs"] = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ############################################################################
    # train
    
    # Figure out how many steps we should save the Accelerator states
    if hasattr(args.checkpointing_steps, "isdigit"):
        checkpointing_steps = args.checkpointing_steps
        if args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
    else:
        checkpointing_steps = None

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg['optimizer']['num_train_epochs']}")
    logger.info(f"  Instantaneous batch size per device = {cfg['optimizer']['per_device_train_batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg['optimizer']['gradient_accumulation_steps']}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)    
    completed_steps = 0
    cfg["output_dir"] = f"{cfg["output_dir"]}/{cfg["exp_name"]}"
    for epoch in range(0, cfg["optimizer"]["num_train_epochs"]):
        start_time = time.time()
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            # print(outputs)
            loss = outputs["loss"]
            cur_loss = loss.item()
            # logger.info(f"train {loss}, {loss.dtype}")
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()    
            # print(loss.detach().float())
            loss = loss / cfg["optimizer"]["gradient_accumulation_steps"]
            accelerator.backward(loss)
            if step % cfg["optimizer"]["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                accelerator.log(
                    {
                        "cur_loss": cur_loss,
                        "step": completed_steps,
                    },
                    # step=completed_steps,
                )
                # progress_bar.set_postfix({'loss': total_loss/step})
                cur_lr = lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix({'lr': f"{cur_lr:.6f}", 'loss': total_loss/step})
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        eval_loss, all_sample, all_prompt = eval_model(model, dev_dataloader, accelerator, logger, tokenizer=custom_collate.tokenizer)
                        # print(eval_loss)
                        all_save = [wandb.Image(p, caption=c) for p, c in zip(all_sample, all_prompt)]
                        if args.with_tracking:
                            accelerator.log(
                                {
                                    "examples": all_save,
                                    "eval_loss": eval_loss,
                                    "train_loss": total_loss.item() / step,
                                    "epoch": epoch,
                                    "step": completed_steps,
                                },
                                # step=completed_steps,
                            )
                        output_dir = f"step_{epoch}_{completed_steps}"
                        if cfg["output_dir"] is not None:
                            output_dir = os.path.join(cfg["output_dir"], output_dir)
                    # accelerator.save_state(output_dir)
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        # save_dir = "test"
                        unwrapped_model.llm.model.save_pretrained(
                            f"{output_dir}/llm", is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                        accelerator.save_model(unwrapped_model.mlp_image_2_llm, f"{output_dir}/mlp_image_2_llm")
                        accelerator.save_model(unwrapped_model.mlp_llm_2_image, f"{output_dir}/mlp_llm_2_image")
                        torch.cuda.empty_cache()
                        # accelerator.save_model(unwrapped_model.diffloss, f"{output_dir}/diff")
                        model.train()

        output_dir = f"step_{epoch}_{completed_steps}"
        if cfg["output_dir"] is not None:
            output_dir = os.path.join(cfg["output_dir"], output_dir)
    # accelerator.save_state(output_dir)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        # save_dir = "test"
        unwrapped_model.llm.model.save_pretrained(
            f"{output_dir}/llm", is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        accelerator.save_model(unwrapped_model.mlp_image_2_llm, f"{output_dir}/mlp_image_2_llm")
        accelerator.save_model(unwrapped_model.mlp_llm_2_image, f"{output_dir}/mlp_llm_2_image")
        # accelerator.save_model(unwrapped_model.diffloss, f"{output_dir}/diff")


if __name__ == "__main__":
    main()

    