from diffusers import AutoencoderKL
from src.tokenizer import AutoencoderKLEncoder, AutoencoderKLDecoder
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import v2
from diffusers import StableDiffusion3Pipeline
from torchvision.utils import save_image
import yaml
import numpy as np
import torch
import os
from src.metrics import calculate_psnr_pt, calculate_ssim_pt
from src.utils import norm_range
from cosmos_tokenizer.image_lib import ImageTokenizer
from huggingface_hub import login, snapshot_download
import os
from tqdm import tqdm
from diffusers.models.autoencoders.vae import DecoderOutput
# You could get your Hugging Face token from https://huggingface.co/settings/tokens
# login(token=<YOUT-HF-TOKEN>, add_to_git_credential=True)
# You could specify the tokenizers you want to download.
model_names = [
        "Cosmos-Tokenizer-CI8x8",
        "Cosmos-Tokenizer-CI16x16",
        # "Cosmos-Tokenizer-CV4x8x8",
        # "Cosmos-Tokenizer-CV8x8x8",
        # "Cosmos-Tokenizer-CV8x16x16",
        # "Cosmos-Tokenizer-DI8x8",
        # "Cosmos-Tokenizer-DI16x16",
        # "Cosmos-Tokenizer-DV4x8x8",
        # "Cosmos-Tokenizer-DV8x8x8",
        # "Cosmos-Tokenizer-DV8x16x16",
]

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

class DenoiseDataset(Dataset):
    def __init__(self, mode="train"):
        super(DenoiseDataset, self).__init__()
        self.mode = mode
        self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        self.crop = v2.Lambda(lambda pil_image: center_crop_arr(pil_image, 256))
        # self.Norm
        self.norm = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.full_dataset)
    
    def __getitem__(self, index):
        cur_item = self.full_dataset[index]
        img = cur_item["image"].convert("RGB")
        img_256 = self.crop(img)
        img_256_norm = self.norm(img_256)
        return {"pixel_values":img_256_norm}


class SpecificDataset(Dataset):
    def __init__(self, dataname = "CBSD68"):
        super(SpecificDataset, self).__init__()
        # self.full_dataset = load_dataset("ILSVRC/imagenet-1k", split=f"{mode}")
        self.full_dataset = []
        if dataname == "CBSD68":
            for filename in os.listdir("/scratch2/f0072r1/img_dataset/CBSD68-dataset/CBSD68/noisy50"):
                full_path = os.path.join("/scratch2/f0072r1/img_dataset/CBSD68-dataset/CBSD68/original_png", filename)
                img_clean = Image.open(full_path)
                self.full_dataset.append({
                    "image": img_clean,
                })

        self.transform = v2.Compose([
            v2.ToImage(), 
            v2.Resize((256, 256)),
            # v2.RGB(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    
    def __len__(self):
        return len(self.full_dataset)

    def __getitem__(self, index):
        cur_item = self.full_dataset[index]
        # img = cur_item["image_noise"].convert("RGB")
        # img = self.transform(img)
        
        # img_clean = img
        # img_noise = self.noise(img)
        img = cur_item["image"].convert("RGB")
        img = self.transform(img)
        # img_noise = self.norm(img_noise)
        # img_noise = self.norm(img_noise)
        # img_clean = self.norm(img)
        return {"pixel_values":img}

def recon_kl16(data_loader):
    with open("/scratch2/f0072r1/res_gemma/cfgs/mae_gemma_denoise.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    img_encoder = AutoencoderKLEncoder(**cfg["img_tokenizer"])
    img_decoder = AutoencoderKLDecoder(**cfg["img_tokenizer"])
    img_encoder = img_encoder.to("cuda:0")
    img_decoder = img_decoder.to("cuda:0")
    img_encoder.eval()
    img_decoder.eval()

    all_raw_img = []
    all_rec_img = []

    with torch.no_grad():
        for data_iter_step, batch in enumerate(tqdm(data_loader)):
            samples = batch['pixel_values']
            samples = samples.to("cuda:0")
            raw_unnorm = norm_range(samples, (-1,1))
            posterior = img_encoder(samples)
            img_tokens = img_encoder.patchify(posterior.sample())
            img_rec = img_decoder(img_tokens)
            pred_unnorm = norm_range(img_rec, (-1,1))
            # psnr = calculate_psnr_pt(pred_unnorm, raw_unnorm, crop_border=4)
            # ssim = calculate_ssim_pt(pred_unnorm, raw_unnorm, crop_border=4)
            all_raw_img.append(raw_unnorm)
            all_rec_img.append(pred_unnorm)
            if data_iter_step > 8:
                break
    
    all_raw_img = torch.cat(all_raw_img, dim=0)
    all_rec_img = torch.cat(all_rec_img, dim=0)
    return all_raw_img, all_rec_img

def recon_sd35(data_loader):
    vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/stable-diffusion-3.5-large/vae")
    vae = vae.to("cuda:0")
    vae.eval()

    all_raw_img = []
    all_rec_img = []

    with torch.no_grad():
        for data_iter_step, batch in enumerate(tqdm(data_loader)):
            samples = batch['pixel_values']
            samples = samples.to("cuda:0")
            raw_unnorm = norm_range(samples, (-1,1))
            posterior = vae.encode(samples).latent_dist
            img_latent = posterior.sample()
            img_rec = vae.decode(img_latent).sample        
            pred_unnorm = norm_range(img_rec, (-1,1))
            all_raw_img.append(raw_unnorm)
            all_rec_img.append(pred_unnorm)
            if data_iter_step > 8:
                break
    
    all_raw_img = torch.cat(all_raw_img, dim=0)
    all_rec_img = torch.cat(all_rec_img, dim=0)
    return all_raw_img, all_rec_img

def recon_cosmos(data_loader):
    model_name = "Cosmos-Tokenizer-CI8x8"
    # input_tensor = torch.randn(1, 3, 256, 256).to('cuda').to(torch.bfloat16)  # [B, C, H, W]
    encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
    decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')
    
    # encoder = encoder.to("cuda:0").to(torch.float32)
    # decoder = decoder.to("cuda:0").to(torch.float32)
    encoder = encoder.to("cuda:0") # .to(torch.float32)
    decoder = decoder.to("cuda:0") # .to(torch.float32)
    for param in encoder.parameters():
        print("encoder type", param.dtype)
        break
    for param in decoder.parameters():
        print("decoder type", param.dtype)
        break
    encoder.eval()
    decoder.eval()

    all_raw_img = []
    all_rec_img = []

    with torch.no_grad():
        for data_iter_step, batch in enumerate(tqdm(data_loader)):
            samples = batch['pixel_values']
            samples = samples.to("cuda:0") # .to(torch.float32)
            print("samples", samples.dtype)
            raw_unnorm = norm_range(samples, (-1,1))
            # posterior = vae.encode(samples).latent_dist
            (latent,) = encoder.encode(samples)
            print("latent", latent.shape, latent.dtype)
            # img_latent = posterior.sample()
            # img_rec = vae.decode(img_latent).sample        
            img_rec = decoder.decode(latent)
            pred_unnorm = norm_range(img_rec, (-1,1))
            all_raw_img.append(raw_unnorm)
            all_rec_img.append(pred_unnorm)
            if data_iter_step > 8:
                break
    
    all_raw_img = torch.cat(all_raw_img, dim=0)
    all_rec_img = torch.cat(all_rec_img, dim=0)
    return all_raw_img, all_rec_img

if __name__ == "__main__":
    save_img = True
    # dataset_train = SpecificDataset(dataname="CBSD68")
    dataset_train = DenoiseDataset(mode="validation")
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        num_workers=8,
        shuffle=False,
        drop_last=False,  # Don't drop in cache
    )
    # all_raw_img, all_rec_img = recon_kl16(data_loader=data_loader)
    all_raw_img, all_rec_img = recon_sd35(data_loader=data_loader)
    # all_raw_img, all_rec_img = recon_cosmos(data_loader=data_loader)
    # exit(0)
    psnr = calculate_psnr_pt(all_raw_img, all_rec_img, crop_border=4)
    ssim = calculate_ssim_pt(all_raw_img, all_rec_img, crop_border=4)
    if save_img == True:
        save_fol = "output_imgs/test_ae/"
        os.makedirs(save_fol, exist_ok=True)
        for img_id in range(all_raw_img.shape[0]):
            save_image(all_raw_img[img_id:img_id + 1, :,:,:], f"{save_fol}/img_{img_id}_raw.png", normalize=False)
            save_image(all_rec_img[img_id:img_id + 1, :,:,:], f"{save_fol}/img_{img_id}_rec.png", normalize=False)

    print("PSNR: ", torch.mean(psnr))
    print("SSIM: ", torch.mean(ssim))
    # print(psnr)
    # print(ssim)
    # vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/stable-diffusion-3.5-large/vae")
    # vae = vae.to("cuda:0")

    # model_name = "Cosmos-Tokenizer-CI8x8"
    # # input_tensor = torch.randn(1, 3, 256, 256).to('cuda').to(torch.bfloat16)  # [B, C, H, W]
    # encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
    # decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')
    # encoder = encoder.to("cuda:0")
    # decoder = decoder.to("cuda:0")
    # print(encoder)
    # print(decoder)
    # exit(0)
    # reconstructed_tensor = decoder.decode(latent)
    # (latent,) = encoder.encode(input_tensor)
    # print("log", latent.shape)

    # with open("/scratch2/f0072r1/res_gemma/cfgs/mae_gemma_denoise.yaml") as f:
    #     cfg = yaml.load(f, Loader=yaml.FullLoader)
    # img_encoder = AutoencoderKLEncoder(**cfg["img_tokenizer"])
    # img_decoder = AutoencoderKLDecoder(**cfg["img_tokenizer"])
    # img_encoder = img_encoder.to("cuda:0")
    # img_decoder = img_decoder.to("cuda:0")


    # save_fol = "test_vae_kl16_2"
    # os.makedirs(save_fol, exist_ok=True)
    # all_psnr = []
    # all_ssim = []
    # with torch.no_grad():
    #     for data_iter_step, batch in enumerate(tqdm(data_loader_train)):
    #         samples = batch['pixel_values']
    #         samples = samples.to("cuda:0")
    #         raw_unnorm = norm_range(samples, (-1,1))
    #         # all_raw.append(samples)
    #         # print(samples.shape)
    #         # samples = samples.to(device, non_blocking=True)
    #         # save_image(samples, f"{save_fol}/tmp{data_iter_step}.png", nrow=int(4), normalize=True, value_range=(-1, 1))
            
    #             # posterior = img_encoder.encode(samples).latent_dist
    #         posterior = img_encoder(samples)
    #             # (latent,) = encoder.encode(samples)
    #         # print(posterior)
    #         # x = posterior.sample()
    #         img_tokens = img_encoder.patchify(posterior.sample())
    #         # print("log latent shape end", img_tokens.shape)
    #         # exit(0)
    #             # img_rec = decoder.decode(latent)
    #         img_rec = img_decoder(img_tokens)
    #         pred_unnorm = norm_range(img_rec, (-1,1))
    #         psnr = calculate_psnr_pt(pred_unnorm, raw_unnorm, crop_border=4)
    #         ssim = calculate_ssim_pt(pred_unnorm, raw_unnorm, crop_border=4)
    #         # print(type(psnr), psnr.shape)
    #         # print(type(ssim), ssim.shape)
    #         all_psnr.append(psnr)
    #         all_ssim.append(ssim)
    #         if data_iter_step > 100:
    #             break
    #     # if data_iter_step > 5:
    #     #     break
    #     # all_reconstructed.append(img_rec)
    #     # print(img_rec.shape)
    #     # save_image(img_rec, f"{save_fol}/tmp{data_iter_step}_r.png", nrow=int(4), normalize=True, value_range=(-1, 1))
    # all_psnr = torch.cat(all_psnr)
    # all_ssim = torch.cat(all_ssim)
    # # print(all_psnr.shape)
    # print(torch.mean(all_psnr))
    # print(torch.mean(all_ssim))
    # for data_iter_step, batch in enumerate(data_loader_train):
    #     samples = batch['pixel_values']
    #     samples = samples.to("cuda:0")
    #     # print(samples.shape)
    #     # samples = samples.to(device, non_blocking=True)
    #     save_image(samples, f"{save_fol}/tmp{data_iter_step}.png", nrow=int(4), normalize=True, value_range=(-1, 1))
    #     with torch.no_grad():
    #         posterior = vae.encode(samples).latent_dist
    #     # print(posterior)
    #     x = posterior.sample()
    #     print("log x shape end", x.shape)
    #     # exit(0)
    #     img_rec = vae.decode(x).sample
    #     print(img_rec.shape)
    #     save_image(img_rec, f"{save_fol}/tmp{data_iter_step}_r.png", nrow=int(4), normalize=True, value_range=(-1, 1))
        # break
        # print(img_rec.shape)
        # save_image(img_rec, f"save_raw/sample_after_{data_iter_step}.png", normalize=True, value_range=(-1, 1))
        # save_image(samples, f"save_raw/sample_before_{data_iter_step}.png", normalize=True, value_range=(-1, 1))
    # posterior = vae.encode(samples)

    # pipe = StableDiffusion3Pipeline.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
    # pipe = pipe.to("cuda")

    # image = pipe(
    #     "A capybara holding a sign that reads Hello World",
    #     num_inference_steps=28,
    #     guidance_scale=3.5,
    # ).images[0]
    # image.save("capybara.png")

    # for model_name in model_names:
    #     hf_repo = "nvidia/" + model_name
    #     local_dir = "pretrained_ckpts/" + model_name
    #     os.makedirs(local_dir, exist_ok=True)
    #     print(f"downloading {model_name} to {local_dir}...")
    #     snapshot_download(repo_id=hf_repo, local_dir=local_dir)

    
