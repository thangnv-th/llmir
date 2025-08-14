import torch
import requests
import safetensors
import logging
import yaml
import numpy as np

import torch.nn as nn

from cosmos_tokenizer.image_lib import ImageTokenizer
from src.utils import norm_range
from transformers import AutoImageProcessor, ViTMAEModel, ViTImageProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma2Model, Gemma2ForCausalLM
from diffusers import AutoencoderKL
from transformers.activations import ACT2FN, PytorchGELUTanh
from transformers.models.vit_mae.modeling_vit_mae import get_2d_sincos_pos_embed
from peft import get_peft_model, LoraConfig, PeftModel
from timm.layers import Mlp
from accelerate.logging import get_logger
from PIL import Image
from pprint import pprint
from src.tokenizer import AutoencoderKLEncoder, AutoencoderKLDecoder
from src.losses import L1Loss, LPIPSLoss, PerceptualLoss, GANLoss
from src.diffloss import DiffLoss
from accelerate.utils import tqdm
from segmentation_models_pytorch.losses import DiceLoss

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger = get_logger(__name__, log_level="INFO")

class VQGANDiscriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_layers=4, model_path=None):
        super().__init__()

        layers = [nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ndf_mult = 1
        ndf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            ndf_mult_prev = ndf_mult
            ndf_mult = min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * ndf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        ndf_mult_prev = ndf_mult
        ndf_mult = min(2 ** n_layers, 8)

        layers += [
            nn.Conv2d(ndf * ndf_mult_prev, ndf * ndf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * ndf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        layers += [
            nn.Conv2d(ndf * ndf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = nn.Sequential(*layers)

        if model_path is not None:
            chkpt = torch.load(model_path, map_location='cpu')
            if 'params_d' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params_d'])
            elif 'params' in chkpt:
                self.load_state_dict(torch.load(model_path, map_location='cpu')['params'])
            else:
                raise ValueError(f'Wrong params!')

    def forward(self, x):
        return self.main(x)
    
class LoRALLM(nn.Module):
    def __init__(self, cfgs, CausalLM=False):
        super().__init__()
        # build llm
        if not CausalLM:
            self.model = Gemma2Model.from_pretrained(
                "hf_w/gemma-2-2b-it",
                # torch_dtype=torch.float16,
            )
        else:
            self.model = Gemma2ForCausalLM.from_pretrained(
                "hf_w/gemma-2-2b-it",
                # torch_dtype=torch.float16,
            )
    
    def build_lora(self, cfg):
        config = LoraConfig(
            r=cfg["llm"]['lora']['rank'],
            target_modules=cfg["llm"]['lora']['target_modules'],
            modules_to_save=cfg["llm"]['lora']['modules_to_save'],
            lora_alpha=cfg["llm"]['lora']['alpha'],
            lora_dropout=cfg["llm"]['lora']['dropout']
        )
        self.model = get_peft_model(self.model, config)
        # self.model = self.model.to(torch.half)
        self.model.print_trainable_parameters()

    def load_lora_weight(self, ckpt_dir, is_trainable=False):
        self.model = PeftModel.from_pretrained(self.model, ckpt_dir, is_trainable=is_trainable)

    def forward(self,inputs):
        # logger.info(f"{self.model.dtype}")
        return self.model(**inputs)

class ImageEncoder(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.model = ViTMAEModel.from_pretrained("hf_w/vit-mae-large")

    def forward(self, inputs):
        outputs = self.model(**inputs)
        # re-order to original
        x = outputs["last_hidden_state"]
        x_ = x[:, 1:, :]
        ids_restore = outputs["ids_restore"] 
        tmp = ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        # print("log in vit", ids_restore.shape, x_.shape, tmp.shape)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)
        return x


class ResLLM(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        # self.img_encoder = ImageEncoder(cfgs)
        self.img_encoder = AutoencoderKLEncoder(**cfgs["img_tokenizer"])
        self.img_decoder = AutoencoderKLDecoder(**cfgs["img_tokenizer"])
        # load llm
        self.llm = LoRALLM(cfgs)
        # print(self.llm.model)
        # pprint(self.image_encoder.config)
        self.mlp_image_2_llm = Mlp(
            in_features=self.img_encoder.token_dims,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=self.img_encoder.token_dims,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.img_encoder.requires_grad_(False)
        self.img_decoder.requires_grad_(False)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.img_encoder.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        self.l2loss = nn.MSELoss()
        self.pixloss = L1Loss()
        # self.perceptual_loss = LPIPSLoss(use_input_norm=True, range_norm=True)

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            # logger.info(f"Load mlp_image_2_llm message {message}")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
            # logger.info(f"Load mlp_llm_2_image message {message}")
        else:
            # logger.info(f"Build new model")
            self.llm.build_lora(cfg)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.img_encoder.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # print("============================== log pos emb", self.decoder_pos_embed.shape)

    def prepare_causal_attention_mask_for_visual(
        self,
        sequence_length: int,
        num_pred_visual_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        num_base_tokens = sequence_length - num_pred_visual_tokens
        causal_mask[num_base_tokens:, num_base_tokens:] = torch.full(
            (num_pred_visual_tokens, num_pred_visual_tokens), fill_value=min_dtype, dtype=dtype, device=device
        ).fill_diagonal_(causal_mask[0,0])
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        return causal_mask

    def forward_loss(self, pred_latent, gt_latent, pred_img=None, gt_imgs=None):
        """
        pred_latent: [N, 1024]
        gt_latent: [N, 1024]
        """
        # l2_latent = self.l2loss(pred_latent.reshape(pred_latent.shape[0] * pred_latent.shape[1], pred_latent.shape[2]), gt_latent.reshape(gt_latent.shape[0] * gt_latent.shape[1], gt_latent.shape[2]))
        l1_img = self.pixloss(pred_img, gt_imgs)
        # percept_loss = self.perceptual_loss(pred_img, gt_imgs)
        # return l2_latent + l1_img + 0.3 * percept_loss
        # return l2_latent + l1_img
        # print("============================== loss", l1_img)
        return l1_img
        # return 

    def forward_llms(self, llm_inputs, pred_patch_id):
        # add token to predict
        all_placeholder_token = []
        for batch_id in range(llm_inputs.shape[0]):
            # print("placeholder_token", self.placeholder_token.shape, pred_patch_id[batch_id].shape[0])
            placeholder_token = self.placeholder_token.repeat(1, pred_patch_id[batch_id].shape[0], 1)
            # print("placeholder_token", placeholder_token.shape)
            positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, pred_patch_id[batch_id])
            # print("positional_embeddings", positional_embeddings.shape)
            # positional_embeddings = positional_embeddings.squeeze().unsqueeze(1)
            placeholder_token = placeholder_token + positional_embeddings
            # print("final", placeholder_token.shape)
            all_placeholder_token.append(placeholder_token)
        all_placeholder_token = torch.cat(all_placeholder_token, dim=0)
        # print("------", all_placeholder_token.shape)
        llm_inputs = torch.cat((llm_inputs, all_placeholder_token), dim=1)
        # print("final inputshape", llm_inputs.shape)
        attention_mask = self.prepare_causal_attention_mask_for_visual(
            sequence_length=llm_inputs.shape[1],
            num_pred_visual_tokens=pred_patch_id.shape[1],
            dtype=llm_inputs.dtype,
            device=llm_inputs.device,
            batch_size=llm_inputs.shape[0]
        )
        # print("====================================================== attention mask", attention_mask.shape)

        llm_output = self.llm({
            "inputs_embeds": llm_inputs,
            "attention_mask": attention_mask,
        })
        num_origin_token = llm_inputs.shape[1] - pred_patch_id.shape[1]
        # print(llm_output["last_hidden_state"].shape)
        return llm_output["last_hidden_state"][:, num_origin_token:, :]
        # llm_output = llm_output["last_hidden_state"][:, -1, :]

    def forward(self, pixel_values, pred_patch_id, gt_latent=None, gt_images=None):
        # pred_patch_id = torch.squeeze(pred_patch_id)
        posterior = self.img_encoder(pixel_values)
        img_tokens = self.img_encoder.patchify(posterior.sample().mul_(0.2325))
        # print("log 1", img_tokens.shape, gt_latent.shape, pred_patch_id.shape)
        
        # map to llm token space
        llm_inputs = self.mlp_image_2_llm(img_tokens)
        # print("log llm_inputs", llm_inputs.shape)

        # we don't need to add positional embeddings because Gemma already add rotary emb
        llm_output = self.forward_llms(llm_inputs, pred_patch_id)
        # print("log after llm", llm_output.shape)
        # print("log 2 llm_inputs", llm_inputs.shape)
        
        # print('log llm output', llm_output.shape)
        img_token_pred = self.mlp_llm_2_image(llm_output)
        # print('log img_token_pred', img_token_pred.shape)
        img_pred = self.img_decoder(img_token_pred / 0.2325)
        # print("log pred_img", img_pred.shape)
        if gt_images is not None or self.training:
            loss = self.forward_loss(
                pred_latent= img_token_pred, 
                gt_latent= gt_latent,
                pred_img=img_pred,
                gt_imgs= gt_images
            )
        else:
            loss = None
        return {
            "pred_token": img_token_pred,
            "pred_img": img_pred,
            "loss": loss,
        }


class ResLLMAE(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        self.img_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit')
        self.img_decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit')
        self.llm = LoRALLM(cfgs)
        self.mlp_image_2_llm = Mlp(
            in_features=16,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=16,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.img_encoder.requires_grad_(False)
        self.img_decoder.requires_grad_(False)

        self.num_patches = 32 * 32
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        self.latent_loss = nn.MSELoss(reduction="mean")
        self.pix_loss = L1Loss()
        self.perceptual_loss = LPIPSLoss(use_input_norm=True, range_norm=True)
        # self.perceptual_loss = PerceptualLoss(
        #     layer_weights={
        #         'conv1_2': 0.1,
        #         'conv2_2': 0.1,
        #         'conv3_4': 1,
        #         'conv4_4': 1,
        #         'conv5_4': 1
        #     },
        #     range_norm=True,
        #     use_input_norm=True,
        #     style_weight=1.0
        # )

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = 1
        h_, w_ = h // p, w // p # 32
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = 1
        c = 16
        h_, w_ = 32, 32
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            # logger.info(f"Load mlp_image_2_llm message {message}")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
            # logger.info(f"Load mlp_llm_2_image message {message}")
        else:
            # logger.info(f"Build new model")
            self.llm.build_lora(cfg)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        # print("============================== log pos emb", self.decoder_pos_embed.shape)

    def prepare_causal_attention_mask_for_visual(
        self,
        sequence_length: int,
        num_pred_visual_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (sequence_length, sequence_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        num_base_tokens = sequence_length - num_pred_visual_tokens
        causal_mask[num_base_tokens:, num_base_tokens:] = torch.full(
            (num_pred_visual_tokens, num_pred_visual_tokens), fill_value=min_dtype, dtype=dtype, device=device
        ).fill_diagonal_(causal_mask[0,0])
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        return causal_mask

    def forward_loss(self, pred_latent, gt_latent, pred_img=None, gt_imgs=None):
        """
        pred_latent: [N, 1024]
        gt_latent: [N, 1024]
        """
        # print("log pred latent", pred_latent.shape)
        # print("log gt latent", gt_latent.shape)
        latent_loss = self.latent_loss(pred_latent.reshape(pred_latent.shape[0] * pred_latent.shape[1], pred_latent.shape[2]), gt_latent.reshape(gt_latent.shape[0] * gt_latent.shape[1], gt_latent.shape[2]))
        pix_loss = self.pix_loss(pred_img, gt_imgs)
        # print("log pix loss", pix_loss)
        # print("log latent loss", latent_loss)
        percep_loss = self.perceptual_loss(pred_img, gt_imgs)
        # print("log percep loss", percep_loss)
        # print("log style loss", style_loss)
        # return l2_latent + l1_img + 0.3 * percept_loss
        # return l2_latent + l1_img
        # print("============================== debug grad_fn", l1_img.grad_fn)
        return pix_loss + 0.5 * latent_loss + 0.3 * percep_loss
        # return 

    def forward_llms(self, llm_inputs, pred_patch_id):
        # print("log pred_id shape", pred_patch_id.shape)
        # add token to predict
        all_placeholder_token = []
        for batch_id in range(llm_inputs.shape[0]):
            placeholder_token = self.placeholder_token.repeat(1, pred_patch_id[batch_id].shape[0], 1)
            positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, pred_patch_id[batch_id])
            placeholder_token = placeholder_token + positional_embeddings
            all_placeholder_token.append(placeholder_token)
        all_placeholder_token = torch.cat(all_placeholder_token, dim=0)
        llm_inputs = torch.cat((llm_inputs, all_placeholder_token), dim=1)
        # print("log llm input", llm_inputs.shape)
        attention_mask = self.prepare_causal_attention_mask_for_visual(
            sequence_length=llm_inputs.shape[1],
            num_pred_visual_tokens=pred_patch_id.shape[1],
            dtype=llm_inputs.dtype,
            device=llm_inputs.device,
            batch_size=llm_inputs.shape[0]
        )
        # print("model1", llm_inputs.dtype)
        # print("model1", attention_mask.dtype)
        llm_output = self.llm({
            "inputs_embeds": llm_inputs,
            "attention_mask": attention_mask,
        })
        num_origin_token = llm_inputs.shape[1] - pred_patch_id.shape[1]
        return llm_output["last_hidden_state"][:, num_origin_token:, :]
        
    def forward(self, pixel_values, pred_patch_id, gt_latent=None, gt_images=None):
        # print("log", pixel_values.shape, pred_patch_id.shape)
        (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
        if gt_images is not None:
            (good_img_tokens,) = self.img_encoder.encode(gt_images)
        
        bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        if gt_images is not None:
            good_img_tokens = good_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)

        # map to llm token space
        bad_img_tokens = self.patchify(bad_img_tokens)
        if gt_images is not None:
            good_img_tokens = self.patchify(good_img_tokens)
        else:
            good_img_tokens = None
        llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
        # print("============================== debug grad_fn llm_inputs", llm_inputs.grad_fn)
        
        # we don't need to add positional embeddings because Gemma already add rotary emb
        llm_output = self.forward_llms(llm_inputs, pred_patch_id)
        # print("============================== debug grad_fn llm_output", llm_output.grad_fn)
        img_token_pred = self.mlp_llm_2_image(llm_output)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred", img_token_pred.grad_fn)
        img_latent = self.unpatchify(img_token_pred)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred after unpatchify", img_token_pred.grad_fn)
        img_latent = img_latent.to(self.img_decoder._dtype)
        # print("============================== debug grad_fn img_token_pred before decode", img_token_pred.grad_fn)
        img_pred = self.img_decoder.decode(img_latent)
        # print("============================== debug grad_fn img_pred", img_pred.grad_fn)
        if gt_images is not None:
            loss = self.forward_loss(
                pred_latent= img_token_pred, 
                gt_latent= good_img_tokens,
                pred_img=img_pred,
                gt_imgs= gt_images
            )
        else:
            loss = None
        
        return {
            "pred_token": img_token_pred,
            "pred_img": img_pred,
            "loss": loss,
        }

class ResLLMAEText(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        # self.img_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit')
        # self.img_decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit')
        self.cfgs = cfgs
        self.img_vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/FLUX.1-dev/vae")
        self.img_vae.requires_grad_(False)
        self.img_vae.eval()

        self.llm = LoRALLM(cfgs)
        self.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        self.mlp_image_2_llm = Mlp(
            in_features=16,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=16,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        # self.img_encoder.requires_grad_(False)
        # self.img_decoder.requires_grad_(False)
        
        

        self.num_patches = 32 * 32
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        # self.latent_loss = nn.MSELoss(reduction="mean")
        self.pix_loss = L1Loss()
        self.perceptual_loss = LPIPSLoss(use_input_norm=True, range_norm=True)
        
        if "gan_opt" in cfgs["loss"]:
            self.ganloss = GANLoss(gan_type=cfgs["loss"]["gan_opt"]["gan_type"])
            self.net_d = VQGANDiscriminator()
        
    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = 1
        h_, w_ = h // p, w // p # 32
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = 1
        c = 16
        h_, w_ = 32, 32
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def init(self, cfg, ckpt_dir = None, net_d_ckpt = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            logger.info(f"Load mlp_image_2_llm {message}")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
            logger.info(f"Load mlp_llm_2_image {message}")
        else:
            self.llm.build_lora(cfg)
        if net_d_ckpt is not None:
            message = safetensors.torch.load_model(self.net_d, net_d_ckpt)
            logger.info(f"Load net_d {message}")
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
    def prepare_causal_attention_mask_for_visual(
        self,
        sequence_length: int, # the length of input sequence, includes both text and img tokens
        num_pred_img: int, # number of placeholder tokens
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        total_len = sequence_length + num_pred_img 
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (total_len, total_len), fill_value=min_dtype, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # causal_mask[sequence_length:, sequence_length:] = torch.full(
        #     (num_pred_img, num_pred_img), fill_value=min_dtype, dtype=dtype, device=device
        # ).fill_diagonal_(causal_mask[0,0])
        # causal_mask[sequence_length-num_pred_img-1 : sequence_length-1, sequence_length-num_pred_img-1 : sequence_length-1] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask[total_len-num_pred_img : total_len, total_len-num_pred_img : total_len] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask[sequence_length-num_pred_img-1 : sequence_length-1, sequence_length-num_pred_img-1 : sequence_length-1] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, num_pred_img), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        causal_mask = causal_mask.clone() 
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask

    def get_last_layer(self):
        return self.mlp_llm_2_image.fc2.weight
    
    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        # print("=========== log in calculate_adaptive_weight", torch.norm(recon_grads), torch.norm(g_grads) + 1e-4, d_weight)
        return d_weight
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight
    
    def forward_loss(self, pred_latent, gt_latent, current_iter, pred_img=None, gt_imgs=None, net_d_pred_fake=None, net_d_pred_fake_detach=None, net_d_pred_real=None):
        """
        pred_latent: [N, 1024]
        gt_latent: [N, 1024]
        """
        # latent_loss = self.latent_loss(pred_latent.reshape(pred_latent.shape[0] * pred_latent.shape[1], pred_latent.shape[2]), gt_latent.reshape(gt_latent.shape[0] * gt_latent.shape[1], gt_latent.shape[2]))
        loss_dict = {}
        pix_loss = self.pix_loss(pred_img, gt_imgs)
        loss_dict["pix_loss"] = pix_loss
        percep_loss = self.perceptual_loss(pred_img, gt_imgs)
        loss_dict["percep_loss"] = percep_loss
        total_loss = pix_loss +  0.5 * percep_loss
        loss_dict["recon_loss"] = total_loss
        # calc gan loss
        # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_using"]:
        if "gan_opt" in self.cfgs["loss"]:
            # net_d_pred_fake = self.net_d(pred_img)
            if self.training == True:
                l_g_gan = self.ganloss(net_d_pred_fake, True, is_disc=False)
                last_layer = self.get_last_layer()
                # d_weight = 1.0
                d_weight = self.calculate_adaptive_weight(total_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                d_weight *= self.adopt_weight(1, current_iter, self.cfgs["loss"]["gan_opt"]["net_d_start_using"])
                d_weight *= self.cfgs["loss"]["gan_opt"]["disc_weight"]
                l_g_gan = d_weight * l_g_gan
                if current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_using"]:
                    loss_dict["gan_loss"] = l_g_gan
                    loss_dict["recon_loss"] += l_g_gan
                else:
                    loss_dict["gan_loss"] = torch.tensor(-1)
                    loss_dict["recon_loss"] += l_g_gan * 0.0

        # calc discriminator loss
        # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_train"]:
        if "gan_opt" in self.cfgs["loss"] and self.training:
            # net_d_pred_real = self.net_d(gt_imgs)
            # net_d_pred_fake_detach = self.net_d(pred_img.detach())
            l_d_real = self.ganloss(net_d_pred_real, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(net_d_pred_real.detach())
            
            
            l_d_fake = self.ganloss(net_d_pred_fake_detach, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(net_d_pred_fake_detach.detach())
            if current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_train"]:
                loss_dict["disc_loss"] = l_d_real + l_d_fake
            else:
                loss_dict["disc_loss"] = (l_d_real + l_d_fake) * 0.0
        return loss_dict
        # return 

    def forward_llms(self, input_ids, attention_masks, img_llm_inputs, img_pred_patch_id):
        text_emb = self.llm.model.embed_tokens(input_ids)
        all_placeholder_token = []
        for batch_id in range(img_llm_inputs.shape[0]):
            placeholder_token = self.placeholder_token.repeat(1, img_pred_patch_id[batch_id].shape[0], 1)
            positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, img_pred_patch_id[batch_id])
            placeholder_token = placeholder_token + positional_embeddings
            all_placeholder_token.append(placeholder_token)
        all_placeholder_token = torch.cat(all_placeholder_token, dim=0)
        llm_inputs = torch.cat((text_emb[:, :-1,:], img_llm_inputs, text_emb[:, -1:,:], all_placeholder_token), dim=1)
        attention_mask = self.prepare_causal_attention_mask_for_visual(
            sequence_length=input_ids.shape[-1] + img_llm_inputs.shape[1],
            num_pred_img=img_pred_patch_id.shape[1],
            attention_mask=attention_masks,
            dtype=llm_inputs.dtype,
            device=llm_inputs.device,
            batch_size=llm_inputs.shape[0]
        )
        llm_output = self.llm({
            "inputs_embeds": llm_inputs,
            "attention_mask": attention_mask,
        })
        num_origin_token = llm_inputs.shape[1] - img_pred_patch_id.shape[1]
        return llm_output["last_hidden_state"][:, num_origin_token:, :]
        
    def forward(self, pixel_values, pred_patch_id, gt_latent=None, gt_images=None, input_ids=None, attention_masks=None, current_iter=None):
        # print("log", pixel_values.shape, pred_patch_id.shape)
        # (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
        with torch.no_grad():
            bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.sample()
            # (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
            # if gt_images is not None:
            #     # (good_img_tokens,) = self.img_encoder.encode(gt_images)
            #     good_img_tokens = self.img_vae.encode(gt_images).latent_dist.mean
        
        bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        # if gt_images is not None:
        #     good_img_tokens = good_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)

        # map to llm token space
        bad_img_tokens = self.patchify(bad_img_tokens)
        # if gt_images is not None:
        #     good_img_tokens = self.patchify(good_img_tokens)
        # else:
        #     good_img_tokens = None
        
        img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
        img_llm_inputs += self.decoder_pos_embed
        # print("============================== debug grad_fn llm_inputs", llm_inputs.grad_fn)
        
        # we don't need to add positional embeddings because Gemma already add rotary emb
        llm_output = self.forward_llms(
            input_ids=input_ids,
            attention_masks=attention_masks,
            img_llm_inputs= img_llm_inputs,
            img_pred_patch_id=pred_patch_id
        )
        # print("============================== debug grad_fn llm_output", llm_output.grad_fn)
        img_token_pred = self.mlp_llm_2_image(llm_output)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred", img_token_pred.grad_fn)
        img_latent = self.unpatchify(img_token_pred)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred after unpatchify", img_token_pred.grad_fn)
        img_latent = img_latent.to(self.img_vae.dtype)
        # img_latent = img_latent.to(self.img_decoder._dtype)
        
        # img_pred = self.img_decoder.decode(img_latent) # img_pred in range [-1, 1]
        # print("============================== debug grad_fn img_token_pred before decode", img_token_pred.grad_fn)
        img_pred = self.img_vae.decode(img_latent).sample # img_pred in range [-1, 1]

        # img_pred = self.img_vae.decode(img_latent).sample
        # print("============================== debug grad_fn img_pred", img_pred.grad_fn)
        # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_using"]:
    
        # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_train"] and gt_images is not None:
        if "gan_opt" in self.cfgs["loss"] and self.training:
            net_d_pred_fake = self.net_d(img_pred)
            net_d_pred_fake_detach = self.net_d(img_pred.clone().detach())
            net_d_pred_real = self.net_d(gt_images)
            # print("logit real", torch.mean(net_d_pred_real, dim=(2,3)))
            # print("logit fake", torch.mean(net_d_pred_fake, dim=(2,3)))
        else:
            net_d_pred_fake = None
            net_d_pred_fake_detach = None
            net_d_pred_real = None

        if gt_images is not None and self.training:
            loss_dict = self.forward_loss(
                pred_latent= img_token_pred,
                gt_latent= None,
                pred_img=img_pred,
                gt_imgs= gt_images,
                current_iter=current_iter,
                net_d_pred_fake = net_d_pred_fake,
                net_d_pred_real = net_d_pred_real,
                net_d_pred_fake_detach = net_d_pred_fake_detach
            )
        else:
            loss_dict = {}
        
        return {
            "pred_token": img_token_pred,
            "pred_img": img_pred,
            "loss": loss_dict,
        }


class ResLLMGan(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        # self.img_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit')
        # self.img_decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit')
        self.cfgs = cfgs
        self.img_vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/FLUX.1-dev/vae")
        self.img_vae.requires_grad_(False)
        self.img_vae.eval()

        self.llm = LoRALLM(cfgs)
        # self.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.llm.model.gradient_checkpointing_enable()
        
        self.mlp_image_2_llm = Mlp(
            in_features=16,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=16,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        # self.img_encoder.requires_grad_(False)
        # self.img_decoder.requires_grad_(False)
        
        self.num_patches = 32 * 32
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        # self.latent_loss = nn.MSELoss(reduction="mean")
    
    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = 1
        h_, w_ = h // p, w // p # 32
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = 1
        c = 16
        h_, w_ = 32, 32
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            logger.info(f"Load mlp_image_2_llm {message}")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
            logger.info(f"Load mlp_llm_2_image {message}")
        else:
            self.llm.build_lora(cfg)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
    def prepare_causal_attention_mask_for_visual(
        self,
        sequence_length: int, # the length of input sequence, includes both text and img tokens
        num_pred_img: int, # number of placeholder tokens
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        total_len = sequence_length + num_pred_img 
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (total_len, total_len), fill_value=min_dtype, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # causal_mask[sequence_length:, sequence_length:] = torch.full(
        #     (num_pred_img, num_pred_img), fill_value=min_dtype, dtype=dtype, device=device
        # ).fill_diagonal_(causal_mask[0,0])
        # causal_mask[sequence_length-num_pred_img-1 : sequence_length-1, sequence_length-num_pred_img-1 : sequence_length-1] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask[total_len-num_pred_img : total_len, total_len-num_pred_img : total_len] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask[sequence_length-num_pred_img-1 : sequence_length-1, sequence_length-num_pred_img-1 : sequence_length-1] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, num_pred_img), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        causal_mask = causal_mask.clone() 
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        return causal_mask

    def get_last_layer(self):
        return self.mlp_llm_2_image.fc2.weight
    
    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        # print("=========== log in calculate_adaptive_weight", torch.norm(recon_grads), torch.norm(g_grads) + 1e-4, d_weight)
        return d_weight
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight
    
    def forward_llms(self, input_ids, attention_masks, img_llm_inputs, img_pred_patch_id):
        text_emb = self.llm.model.embed_tokens(input_ids)
        all_placeholder_token = []
        for batch_id in range(img_llm_inputs.shape[0]):
            placeholder_token = self.placeholder_token.repeat(1, img_pred_patch_id[batch_id].shape[0], 1)
            positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, img_pred_patch_id[batch_id])
            placeholder_token = placeholder_token + positional_embeddings
            all_placeholder_token.append(placeholder_token)
        all_placeholder_token = torch.cat(all_placeholder_token, dim=0)
        llm_inputs = torch.cat((text_emb[:, :-1,:], img_llm_inputs, text_emb[:, -1:,:], all_placeholder_token), dim=1)
        attention_mask = self.prepare_causal_attention_mask_for_visual(
            sequence_length=input_ids.shape[-1] + img_llm_inputs.shape[1],
            num_pred_img=img_pred_patch_id.shape[1],
            attention_mask=attention_masks,
            dtype=llm_inputs.dtype,
            device=llm_inputs.device,
            batch_size=llm_inputs.shape[0]
        )
        llm_output = self.llm({
            "inputs_embeds": llm_inputs,
            "attention_mask": attention_mask,
        })
        num_origin_token = llm_inputs.shape[1] - img_pred_patch_id.shape[1]
        return llm_output["last_hidden_state"][:, num_origin_token:, :]
        
    def forward(self, pixel_values, pred_patch_id, gt_latent=None, gt_images=None, input_ids=None, attention_masks=None, current_iter=None):
        # print("log", pixel_values.shape, pred_patch_id.shape)
        # (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
        with torch.no_grad():
            bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.sample()
            # (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
            # if gt_images is not None:
            #     # (good_img_tokens,) = self.img_encoder.encode(gt_images)
            #     good_img_tokens = self.img_vae.encode(gt_images).latent_dist.mean
        
        bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        # if gt_images is not None:
        #     good_img_tokens = good_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)

        # map to llm token space
        bad_img_tokens = self.patchify(bad_img_tokens)
        # if gt_images is not None:
        #     good_img_tokens = self.patchify(good_img_tokens)
        # else:
        #     good_img_tokens = None
        
        img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
        img_llm_inputs += self.decoder_pos_embed
        # print("============================== debug grad_fn llm_inputs", llm_inputs.grad_fn)
        
        # we don't need to add positional embeddings because Gemma already add rotary emb
        llm_output = self.forward_llms(
            input_ids=input_ids,
            attention_masks=attention_masks,
            img_llm_inputs= img_llm_inputs,
            img_pred_patch_id=pred_patch_id
        )
        # print("============================== debug grad_fn llm_output", llm_output.grad_fn)
        img_token_pred = self.mlp_llm_2_image(llm_output)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred", img_token_pred.grad_fn)
        img_latent = self.unpatchify(img_token_pred)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred after unpatchify", img_token_pred.grad_fn)
        img_latent = img_latent.to(self.img_vae.dtype)
        # img_latent = img_latent.to(self.img_decoder._dtype)
        
        # img_pred = self.img_decoder.decode(img_latent) # img_pred in range [-1, 1]
        # print("============================== debug grad_fn img_token_pred before decode", img_token_pred.grad_fn)
        img_pred = self.img_vae.decode(img_latent).sample # img_pred in range [-1, 1]
        # img_pred = self.img_vae.decode(img_latent).sample
        # print("============================== debug grad_fn img_pred", img_pred.grad_fn)
        # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_using"]:
    
        # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_train"] and gt_images is not None:
        
        return {
            "pred_token": img_token_pred,
            "pred_img": img_pred,
            "gt_img": gt_images
        }

class ResTextSegment(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        # self.img_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit')
        # self.img_decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit')
        self.img_vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/FLUX.1-dev/vae")
        self.llm = LoRALLM(cfgs)
        self.llm.model.gradient_checkpointing_enable()
        self.mlp_image_2_llm = Mlp(
            in_features=16,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=16,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        # self.img_encoder.requires_grad_(False)
        # self.img_decoder.requires_grad_(False)
        self.img_vae.eval()
        self.img_vae.requires_grad_(False)

        self.num_patches = 32 * 32
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        self.segment_loss = DiceLoss(
            mode="binary",
            from_logits=False
        )
    
    def patchify(self, x, p = 1):
        bsz, c, h, w = x.shape
        h_, w_ = h // p, w // p # 32
        x = x.reshape(bsz, c, h_, p, w_, p)
        # x = torch.einsum('nchpwq->nhwcpq', x)
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = torch.einsum('nhwpqc->nhwpqc', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]
    
    def unpatchify(self, x, p = 1, dim = 16, h_ = 32, w_ = 32):
        bsz = x.shape[0]
        c = dim
        # h_, w_ = 32, 32
        x = x.reshape(bsz, h_, w_, c, p, p)
        # x = torch.einsum('nhwcpq->nchpwq', x)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
        else:
            self.llm.build_lora(cfg)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
    def prepare_causal_attention_mask_for_visual(
        self,
        sequence_length: int, # the length of input sequence, includes both text and img tokens
        num_pred_img: int, # number of placeholder tokens
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ):
        total_len = sequence_length + num_pred_img 
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full(
            (total_len, total_len), fill_value=min_dtype, dtype=dtype, device=device
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # causal_mask[sequence_length:, sequence_length:] = torch.full(
        #     (num_pred_img, num_pred_img), fill_value=min_dtype, dtype=dtype, device=device
        # ).fill_diagonal_(causal_mask[0,0])
        causal_mask[sequence_length-num_pred_img-1 : sequence_length-1, sequence_length-num_pred_img-1 : sequence_length-1] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask[total_len-num_pred_img : total_len, total_len-num_pred_img : total_len] = causal_mask[0,0] # bidirectional for image tokens
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        attention_mask = torch.cat([attention_mask, torch.ones((batch_size, num_pred_img), device=attention_mask.device, dtype=attention_mask.dtype)], dim=1)
        causal_mask = causal_mask.clone() 
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                        padding_mask, min_dtype
                    )
        return causal_mask

    def forward_loss(self, pred_img=None, mask=None):
        """
        pred_latent: [N, 1024]
        gt_latent: [N, 1024]
        """
        loss = self.segment_loss(
            y_pred = torch.mean(pred_img, dim = 1),
            y_true = mask
        )
        return loss

    def forward_llms(self, input_ids, attention_masks, img_llm_inputs, img_pred_patch_id):
        text_emb = self.llm.model.embed_tokens(input_ids)
        all_placeholder_token = []
        for batch_id in range(img_llm_inputs.shape[0]):
            placeholder_token = self.placeholder_token.repeat(1, img_pred_patch_id[batch_id].shape[0], 1)
            # positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, img_pred_patch_id[batch_id])
            placeholder_token = placeholder_token + self.decoder_pos_embed
            all_placeholder_token.append(placeholder_token)
        all_placeholder_token = torch.cat(all_placeholder_token, dim=0)
        llm_inputs = torch.cat((text_emb[:, :-1,:], img_llm_inputs, text_emb[:, -1:,:], all_placeholder_token), dim=1)
        attention_mask = self.prepare_causal_attention_mask_for_visual(
            sequence_length=input_ids.shape[-1] + img_llm_inputs.shape[1],
            num_pred_img=img_pred_patch_id.shape[1],
            attention_mask=attention_masks,
            dtype=llm_inputs.dtype,
            device=llm_inputs.device,
            batch_size=llm_inputs.shape[0]
        )
        llm_output = self.llm({
            "inputs_embeds": llm_inputs,
            "attention_mask": attention_mask,
        })
        num_origin_token = llm_inputs.shape[1] - img_pred_patch_id.shape[1]
        return llm_output["last_hidden_state"][:, num_origin_token:, :]
        
    def forward(self, pixel_values, pred_patch_id, gt_latent=None, gt_images=None, input_ids=None, attention_masks=None, mask=None):
        # print("log", pixel_values.shape, pred_patch_id.shape)
        # (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
        with torch.no_grad():
            bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.sample()
            # if gt_images is not None:
            #     # (good_img_tokens,) = self.img_encoder.encode(gt_images)
            #     good_img_tokens = self.img_vae.encode(gt_images).latent_dist.mean
        
        bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        # if gt_images is not None:
        #     good_img_tokens = good_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)

        # map to llm token space
        bad_img_tokens = self.patchify(bad_img_tokens)
        # if gt_images is not None:
        #     good_img_tokens = self.patchify(good_img_tokens)
        # else:
        #     good_img_tokens = None
        
        img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
        img_llm_inputs += self.decoder_pos_embed
        # print("============================== debug grad_fn llm_inputs", llm_inputs.grad_fn)
        
        # we don't need to add positional embeddings because Gemma already add rotary emb
        llm_output = self.forward_llms(
            input_ids=input_ids,
            attention_masks=attention_masks,
            img_llm_inputs= img_llm_inputs,
            img_pred_patch_id=pred_patch_id
        )
        # print("============================== debug grad_fn llm_output", llm_output.grad_fn)
        img_token_pred = self.mlp_llm_2_image(llm_output)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred", img_token_pred.grad_fn)
        img_latent = self.unpatchify(img_token_pred)
        # print("log llm output", img_token_pred.shape)
        # print("============================== debug grad_fn img_token_pred after unpatchify", img_token_pred.grad_fn)
        img_latent = img_latent.to(self.img_vae.dtype)
        # print("============================== debug grad_fn img_token_pred before decode", img_token_pred.grad_fn)
        img_pred = self.img_vae.decode(img_latent).sample
        img_pred = norm_range(img_pred, (-1, 1))
        # print("============================== debug grad_fn img_pred", img_pred.grad_fn)
        if mask is not None:
            loss = self.forward_loss(
                pred_img=img_pred,
                mask= mask
            )
        else:
            loss = None
        
        return {
            "pred_img": img_pred,
            "loss": loss,
        }

class ResLLMDiff(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        # self.img_encoder = ImageEncoder(cfgs)
        self.img_encoder = AutoencoderKLEncoder(**cfgs["img_tokenizer"])
        # load llm
        self.llm = LoRALLM(cfgs)
        # print(self.llm.model)
        # pprint(self.image_encoder.config)
        self.mlp_image_2_llm = Mlp(
            in_features=self.img_encoder.token_dims,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 2.0),
            act_layer=PytorchGELUTanh
        )
        self.diffusion_batch_mul = cfgs["diffusion"]["diffusion_batch_mul"]
        # self.mlp_llm_2_image = Mlp(
        #     in_features=self.llm.model.config.hidden_size,
        #     out_features=self.img_encoder.token_dims,
        #     hidden_features=int(self.llm.model.config.hidden_size * 2.0),
        #     act_layer=PytorchGELUTanh
        # )
        self.diffloss = DiffLoss(
            target_channels=self.img_encoder.token_dims,
            z_channels=self.llm.model.config.hidden_size,
            width=cfgs['diffusion']['diffloss_w'],
            depth=cfgs['diffusion']['diffloss_d'],
            num_sampling_steps=cfgs['diffusion']['num_sampling_steps'],
            grad_checkpointing=cfgs['diffusion']['grad_checkpointing']
        )
        self.img_encoder.requires_grad_(False)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.img_encoder.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        self.lossfn = nn.MSELoss()

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            logger.info(f"Load mlp_image_2_llm message {message}")
            message = safetensors.torch.load_model(self.diffloss, f"{ckpt_dir}/diff/model.safetensors")
            logger.info(f"Load diffloss message {message}")
        else:
            logger.info(f"Build new model")
            self.llm.build_lora(cfg)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.img_encoder.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

    # def forward_loss(self, pred_latent, gt_latent):
    #     """
    #     pred_latent: [N, 1024]
    #     gt_latent: [N, 1024]
    #     """
        
    #     return self.lossfn(pred_latent, gt_latent)

    def forward_loss(self, z, target):
        bsz, c_dims = target.shape
        target = target.repeat(self.diffusion_batch_mul, 1)
        # print("log before", target.shape)
        z = z.repeat(self.diffusion_batch_mul, 1)
        # print("log before", z.shape)
        loss = self.diffloss(z=z, target=target)
        return loss
    
    def forward(self, pixel_values, pred_patch_id, gt_latent=None, img_tokens=None, eval=False):
        if eval==True:
            return self.sample_tokens(pixel_values=pixel_values)
        # tokenize images into a sequence of tokens
        pred_patch_id = torch.squeeze(pred_patch_id)
        posterior = self.img_encoder(pixel_values)
        img_tokens = self.img_encoder.patchify(posterior.sample().mul_(0.2325))
        # print("finish tokenize")
        # map images tokens to llm tokens
        llm_inputs = self.mlp_image_2_llm(img_tokens)

        # we don't need to add positional embeddings because Gemma already add rotary emb

        # add placeholder token with positional embeddings
        placeholder_token = self.placeholder_token.repeat(llm_inputs.shape[0], 1, 1)
        positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, pred_patch_id)
        positional_embeddings = positional_embeddings.squeeze().unsqueeze(1)
        placeholder_token = placeholder_token + positional_embeddings
        llm_inputs = torch.cat((llm_inputs, placeholder_token), dim=1)
        # print("finish prepare input")
        # llm forward, get the last hidden vector of placeholder tokens
        llm_output = self.llm({"inputs_embeds": llm_inputs})
        diff_condition = llm_output["last_hidden_state"][:, -1, :]
        # print("log 2 llm_inputs", llm_inputs.shape)
        # print("finish llms")
        # diffusion with the last hidden vector as the conditional vector
        # print("log before diff", diff_condition.shape, gt_latent.shape)
        # loss = self.diffloss(z=diff_condition, target=gt_latent)
        loss = self.forward_loss(z=diff_condition, target=gt_latent)
        output = {
            "loss": loss
        }
        # print("finish diffusion")
        return output

    def sample_tokens(self, pixel_values):
        posterior = self.img_encoder(pixel_values)
        img_tokens = self.img_encoder.patchify(posterior.sample().mul_(0.2325))
        llm_inputs = self.mlp_image_2_llm(img_tokens)
        ret = []
        for id in tqdm(range(self.img_encoder.num_patches)):
            placeholder_token = self.placeholder_token.repeat(llm_inputs.shape[0], 1, 1)
            pred_patch_id = torch.tensor([id], device=llm_inputs.device).repeat(llm_inputs.shape[0])
            positional_embeddings = torch.index_select(self.decoder_pos_embed, 1, pred_patch_id)
            positional_embeddings = positional_embeddings.squeeze().unsqueeze(1)
            placeholder_token = placeholder_token + positional_embeddings
            llm_inputs = torch.cat((llm_inputs, placeholder_token), dim=1)
            llm_output = self.llm({"inputs_embeds": llm_inputs})
            diff_condition = llm_output["last_hidden_state"][:, -1, :]
            # print(diff_condition.shape)
            sampled_token_latent = self.diffloss.sample(diff_condition, temperature=0.1, cfg=1.0)
            # print(sampled_token_latent.shape)
            ret.append(sampled_token_latent.unsqueeze(1))
            # print(ret[-1].shape)
        ret = torch.cat(ret, dim=1)
        return ret

class ResLLMDiffusion(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        
        self.cfgs = cfgs
        if cfgs["img_tokenizer"]["type"] == "VAE":
            self.img_vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/FLUX.1-dev/vae")
            self.img_vae.requires_grad_(False)
            self.img_vae.eval()
        elif cfgs["img_tokenizer"]["type"] == "AE":
            self.img_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit')
            self.img_decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit')
            self.img_encoder.requires_grad_(False)
            self.img_decoder.requires_grad_(False)
        elif cfgs["img_tokenizer"]["type"] == "KL16":
            self.img_encoder = AutoencoderKLEncoder(**cfg["img_tokenizer"])
            self.img_decoder = AutoencoderKLDecoder(**cfg["img_tokenizer"])
            self.img_encoder.requires_grad_(False)
            self.img_decoder.requires_grad_(False)
            self.img_encoder.eval()
            self.img_decoder.eval()

        self.scale_factor = 0.2325        
        self.patch_group = 1
        self.vae_dim = 16
        self.num_patches = int(16 / self.patch_group) * int(16 / self.patch_group)
        self.llm = LoRALLM(cfgs, CausalLM=True)
        
        self.mlp_image_2_llm = Mlp(
            in_features=self.vae_dim * self.patch_group * self.patch_group,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=self.vae_dim * self.patch_group * self.patch_group,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        # self.latent_loss = nn.MSELoss(reduction="mean")
        # if "pixel_opt" in cfgs["loss"]:
        #     self.pix_loss = L1Loss()
        # if "perceptual_opt" in cfgs["loss"]:
        #     self.perceptual_loss = LPIPSLoss(use_input_norm=True, range_norm=True)
        self.diffloss = DiffLoss(
            target_channels=self.vae_dim,
            z_channels=self.llm.model.config.hidden_size,
            width=1024,
            depth=6,
            num_sampling_steps="100",
            grad_checkpointing=True)
    
    def tokenize_image(self, pixel_values):
        with torch.no_grad():
            if self.cfgs["img_tokenizer"]["type"] == "VAE":
                img_tokens = self.img_vae.encode(pixel_values).latent_dist.mean
            elif self.cfgs["img_tokenizer"]["type"] == "AE":
                (img_tokens,) = self.img_encoder.encode(pixel_values)
            elif self.cfgs["img_tokenizer"]["type"] == "KL16":
                posterior = self.img_encoder(pixel_values)
                img_tokens = posterior.sample().mul_(self.scale_factor)
            img_tokens.requires_grad_(False)
            img_tokens = img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        return img_tokens

    def detokenize_image(self, img_latent):
        if self.cfgs["img_tokenizer"]["type"] == "AE":
            img_latent = img_latent.to(self.img_decoder._dtype)
            img_pred = self.img_decoder.decode(img_latent)
        elif self.cfgs["img_tokenizer"]["type"] == "VAE":
            img_latent = img_latent.to(self.img_vae.dtype)
            img_pred = self.img_vae.decode(img_latent).sample
        elif self.cfgs["img_tokenizer"]["type"] == "KL16":
            img_latent = img_latent.to(self.img_decoder.dtype)
            img_pred = self.img_decoder(img_latent / self.scale_factor)
        return img_pred

    def patchify(self, x, p):
        bsz, c, h, w = x.shape
        h_, w_ = h // p, w // p # 32
        x = x.reshape(bsz, c, h_, p, w_, p)
        # x = torch.einsum('nchpwq->nhwcpq', x)
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = torch.einsum('nhwpqc->nhwpqc', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x, h_, w_  # [n, l, d]

    def unpatchify(self, x, p, dim, h_, w_):
        bsz = x.shape[0]
        c = dim
        # h_, w_ = 32, 32
        x = x.reshape(bsz, h_, w_, p, p, c)
        # x = torch.einsum('nhwcpq->nchpwq', x)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            logger.info(f"Load mlp_image_2_llm {message}")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
            logger.info(f"Load mlp_llm_2_image {message}")
        else:
            if cfg["llm"]["train"]:
                self.llm.build_lora(cfg)
                self.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                message = safetensors.torch.load_model(self.mlp_image_2_llm, "/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-lora32-stage-1-ae/step_0_1400/mlp_image_2_llm/model.safetensors")
                logger.info(f"Load mlp_image_2_llm {message}")
                message = safetensors.torch.load_model(self.mlp_llm_2_image, "/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-lora32-stage-1-ae/step_0_1400/mlp_llm_2_image/model.safetensors")
                logger.info(f"Load mlp_llm_2_image {message}")
            else:
                self.llm.model.requires_grad_(False)
        
        # decoder_pos_embed = get_2d_sincos_pos_embed(
        #     self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), add_cls_token=False
        # )
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
    def prepare_causal_attention_mask_for_visual(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_pred_img: int, 
        num_add_img: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        sequence_length = input_ids.shape[1]
        total_len = sequence_length + num_pred_img + num_add_img
        min_dtype = torch.finfo(dtype).min
        indices = (input_ids == 7).nonzero()
        mul = int(indices.shape[0] / input_ids.shape[0])
        # print("log mul", mul)
        attn_masks = []
        boi_pos = []
        for id in range(input_ids.shape[0]):
            # print(input_ids[id])
            causal_mask = torch.full(
                (total_len, total_len), fill_value=min_dtype, dtype=dtype, device=device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            start = indices[id * mul][-1] + 1
            boi_pos.append(start - 1)
            causal_mask[start:start + num_add_img,start:start + num_add_img] = causal_mask[0][0]
            if num_pred_img > 0:
                causal_mask[total_len - num_pred_img:total_len, total_len - num_pred_img:total_len] = causal_mask[0,0]
            # print(start)
            causal_mask[:, :start-2] = min_dtype
            attn_masks.append(torch.unsqueeze(causal_mask, 0))
        attn_masks = torch.stack(attn_masks)
        return attn_masks, boi_pos

    def get_last_layer(self):
        return self.mlp_llm_2_image.fc2.weight
    
    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        # print("=========== log in calculate_adaptive_weight", torch.norm(recon_grads), torch.norm(g_grads) + 1e-4, d_weight)
        return d_weight
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight
    
    def forward_t2i(self, input_ids, attention_masks, img_llm_inputs):
        text_emb = self.llm.model.get_input_embeddings()(input_ids)
        attention_masks_4d, boi_pos = self.prepare_causal_attention_mask_for_visual(
            input_ids=input_ids,
            attention_mask=attention_masks,
            num_add_img=self.num_patches,
            num_pred_img=self.num_patches,
            device=text_emb.device,
            dtype=text_emb.dtype
        )
        placeholder_token = self.placeholder_token.repeat(1, self.num_patches, 1)
        placeholder_token = placeholder_token.repeat(input_ids.shape[0], 1, 1)
        input_embs = []
        for id in range(img_llm_inputs.shape[0]):
            boi_id = boi_pos[id]
            input_emb = torch.cat((text_emb[id, :boi_id+1,:], img_llm_inputs[id], text_emb[id, boi_id+1:,:], placeholder_token[id]), dim=0)
            input_embs.append(input_emb)
        input_embs = torch.stack(input_embs)
        llm_output = self.llm({
            "inputs_embeds": input_embs,
            "attention_mask": attention_masks_4d,
            "feature_only": True
        })
        return llm_output["last_hidden_state"][:, -self.num_patches:, :]
    
    def forward_loss_t2i(self, pred_img=None, gt_img=None):
        """
        pred_latent: [N, 1024]
        gt_latent: [N, 1024]
        """
        ret = {}
        
        return ret
    
    def image_generate(self, pixel_values=None, input_ids_t2i=None, attention_masks_t2i=None):
        with torch.no_grad():
            # bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.mean
            # bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
            bad_img_tokens = self.tokenize_image(pixel_values=pixel_values)
            bad_img_tokens, h_, w_ = self.patchify(bad_img_tokens, self.patch_group)
            img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
            # img_llm_inputs += self.decoder_pos_embed
            llm_output_t2i = self.forward_t2i(
                input_ids=input_ids_t2i,
                attention_masks=attention_masks_t2i,
                img_llm_inputs=img_llm_inputs
            )
            img_token_pred = self.mlp_llm_2_image(llm_output_t2i)
            img_latent = self.unpatchify(img_token_pred, p=self.patch_group, dim=self.vae_dim, h_=h_, w_=w_)
            img_pred = self.detokenize_image(img_latent=img_latent)
            # img_latent = img_latent.to(self.img_vae.dtype)
            # img_pred = self.img_vae.decode(img_latent).sample # img_pred in range [-1, 1]
        return img_pred
        
    def forward(self, pixel_values=None, target_image=None, input_ids_t2i=None, attention_masks_t2i=None, input_ids_i2t=None, attention_masks_i2t=None, labels=None, gen_text_only = False, gen_image_only = False):
        if gen_image_only == True:
            return self.image_generate(
                pixel_values=pixel_values,
                input_ids_t2i=input_ids_t2i,
                attention_masks_t2i=attention_masks_t2i
            )
        if gen_text_only == True:
            return self.text_generate(
                pixel_values=pixel_values,
                input_ids=input_ids_t2i,
                attention_masks=attention_masks_t2i
            )
        bad_img_tokens = self.tokenize_image(pixel_values=pixel_values)
        # with torch.no_grad():
        #     bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.mean
        # bad_img_tokens.requires_grad_(False)
        #     # (bad_img_tokens,) = self.img_encoder.encode(pixel_values)
        #     # if gt_images is not None:
        #     #     # (good_img_tokens,) = self.img_encoder.encode(gt_images)
        #     #     good_img_tokens = self.img_vae.encode(gt_images).latent_dist.mean
        
        # bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        # if gt_images is not None:
        #     good_img_tokens = good_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)

        # map to llm token space
        bad_img_tokens, h_, w_ = self.patchify(bad_img_tokens, self.patch_group)
        # if gt_images is not None:
        #     good_img_tokens = self.patchify(good_img_tokens)
        # else:
        #     good_img_tokens = None
        
        img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
        # img_llm_inputs += self.decoder_pos_embed
        # print("============================== debug grad_fn llm_inputs", llm_inputs.grad_fn)
        
        # we don't need to add positional embeddings because Gemma already add rotary emb
        # llm_output = self.forward_llms(
        #     input_ids=input_ids,
        #     attention_masks=attention_masks,
        #     img_llm_inputs= img_llm_inputs,
        #     img_pred_patch_id=pred_patch_id
        # )
        llm_output_t2i = self.forward_t2i(
            input_ids=input_ids_t2i,
            attention_masks=attention_masks_t2i,
            img_llm_inputs=img_llm_inputs
        )
        # print("============================== debug grad_fn llm_output", llm_output.grad_fn)
        # print("llm_output_t2i", llm_output_t2i.shape)
        img_token_pred = self.mlp_llm_2_image(llm_output_t2i)
        # print("img_token_pred", img_token_pred.shape)
        img_latent = self.unpatchify(img_token_pred, p=self.patch_group, dim=self.vae_dim, h_=h_, w_=w_)
        # img_latent = img_latent.to(self.img_vae.dtype)
        # # print("img_latent", img_latent.shape)
        # img_pred = self.img_vae.decode(img_latent).sample # img_pred in range [-1, 1]
        img_pred = self.detokenize_image(img_latent=img_latent)
        loss_dict = self.forward_loss_t2i(pred_img=img_pred, gt_img=target_image)
        loss_dict["pred_imgs"] = img_pred 
        if input_ids_i2t is not None:
            llm_output_i2t = self.forward_i2t(
                input_ids=input_ids_i2t,
                attention_masks=attention_masks_i2t,
                img_llm_inputs=img_llm_inputs,
                labels=labels
            )
            loss_dict["total_text_loss"] = llm_output_i2t["loss"] 
        # print("img_pred", img_pred.shape)
        # print()
        # print()
        # print(llm_output_t2i)
        # print("llm_output_i2t")
        # print(llm_output_i2t)
        return loss_dict

class ResLLM2Heads(nn.Module):
    def __init__(self, cfgs) -> None:
        super().__init__()
        # build image encoder
        
        self.cfgs = cfgs
        if cfgs["img_tokenizer"]["type"] == "VAE":
            self.img_vae = AutoencoderKL.from_pretrained("/scratch2/f0072r1/res_gemma/hf_w/FLUX.1-dev/vae")
            self.img_vae.requires_grad_(False)
            self.img_vae.eval()
        elif cfgs["img_tokenizer"]["type"] == "AE":
            self.img_encoder = ImageTokenizer(checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/encoder.jit')
            self.img_decoder = ImageTokenizer(checkpoint_dec=f'pretrained_ckpts/Cosmos-Tokenizer-CI8x8/decoder.jit')
            # self.img_encoder = self.img_encoder.to(torch.float)
            # self.img_decoder = self.img_encoder.to(torch.float)
            self.img_encoder.requires_grad_(False)
            self.img_decoder.requires_grad_(False)
            # print("log intint self.img_decoder._dtype", self.img_decoder._dtype)
        self.patch_group = 2
        self.vae_dim = 16
        self.num_patches = int(32 / self.patch_group) * int(32 / self.patch_group)
        self.llm = LoRALLM(cfgs, CausalLM=False)
        
        self.mlp_image_2_llm = Mlp(
            in_features=self.vae_dim * self.patch_group * self.patch_group,
            out_features=self.llm.model.config.hidden_size,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        self.mlp_llm_2_image = Mlp(
            in_features=self.llm.model.config.hidden_size,
            out_features=self.vae_dim * self.patch_group * self.patch_group,
            hidden_features=int(self.llm.model.config.hidden_size * 4.0),
            act_layer=PytorchGELUTanh
        )
        
        
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.llm.model.config.hidden_size), requires_grad=False
        )  # fixed sin-cos embedding # may not use now
        self.placeholder_token = nn.Parameter(torch.zeros(1, 1, self.llm.model.config.hidden_size), requires_grad=False)
        # self.latent_loss = nn.MSELoss(reduction="mean")
        if "pixel_opt" in cfgs["loss"]:
            self.pix_loss = L1Loss()
        if "perceptual_opt" in cfgs["loss"]:
            self.perceptual_loss = LPIPSLoss(use_input_norm=True, range_norm=True)
        if "gan_opt" in cfgs["loss"]:
            self.ganloss = GANLoss(gan_type=cfgs["loss"]["gan_opt"]["gan_type"])
            self.net_d = VQGANDiscriminator()
        # if "caption_opt" in cfgs["loss"]:
        #     self.cap_net = 
    
    def tokenize_image(self, pixel_values):
        # print("log tokenizer", pixel_values.dtype)
        with torch.no_grad():
            if self.cfgs["img_tokenizer"]["type"] == "VAE":
                img_tokens = self.img_vae.encode(pixel_values).latent_dist.mean
            elif self.cfgs["img_tokenizer"]["type"] == "AE":
                (img_tokens,) = self.img_encoder.encode(pixel_values)
            img_tokens.requires_grad_(False)
            img_tokens = img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
        return img_tokens

    def detokenize_image(self, img_latent):
        if self.cfgs["img_tokenizer"]["type"] == "AE":
            # print("log self.img_decoder._dtype", self.img_decoder._dtype)
            img_latent = img_latent.to(self.img_decoder._dtype)
            img_pred = self.img_decoder.decode(img_latent)
        elif self.cfgs["img_tokenizer"]["type"] == "VAE":
            img_latent = img_latent.to(self.img_vae.dtype)
            img_pred = self.img_vae.decode(img_latent).sample
        return img_pred

    def patchify(self, x, p):
        bsz, c, h, w = x.shape
        h_, w_ = h // p, w // p # 32
        x = x.reshape(bsz, c, h_, p, w_, p)
        # x = torch.einsum('nchpwq->nhwcpq', x)
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = torch.einsum('nhwpqc->nhwpqc', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x, h_, w_  # [n, l, d]

    def unpatchify(self, x, p, dim, h_, w_):
        bsz = x.shape[0]
        c = dim
        # h_, w_ = 32, 32
        x = x.reshape(bsz, h_, w_, p, p, c)
        # x = torch.einsum('nhwcpq->nchpwq', x)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]

    def init(self, cfg, ckpt_dir = None, is_trainable=False):
        if ckpt_dir is not None:
            logger.info(f"Load model from {ckpt_dir}")
            self.llm.load_lora_weight(f"{ckpt_dir}/llm", is_trainable=is_trainable)
            message = safetensors.torch.load_model(self.mlp_image_2_llm, f"{ckpt_dir}/mlp_image_2_llm/model.safetensors")
            logger.info(f"Load mlp_image_2_llm {message}")
            message = safetensors.torch.load_model(self.mlp_llm_2_image, f"{ckpt_dir}/mlp_llm_2_image/model.safetensors")
            logger.info(f"Load mlp_llm_2_image {message}")
        else:
            if cfg["llm"]["train"]:
                self.llm.build_lora(cfg)
                self.llm.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                # message = safetensors.torch.load_model(self.mlp_image_2_llm, "/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-lora32-stage-1-ae/step_0_1400/mlp_image_2_llm/model.safetensors")
                # self.mlp_image_2_llm.requires_grad_(False)
                # logger.info(f"Load mlp_image_2_llm {message}")
                # message = safetensors.torch.load_model(self.mlp_llm_2_image, "/scratch2/f0072r1/res_gemma/logs/vae-gemma2-2b-lora32-stage-1-ae/step_0_1400/mlp_llm_2_image/model.safetensors")
                # self.mlp_llm_2_image.requires_grad_(False)
                # logger.info(f"Load mlp_llm_2_image {message}")
            else:
                self.llm.model.requires_grad_(False)
        
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), add_cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
    
    def prepare_causal_attention_mask_for_visual(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_pred_img: int, 
        num_add_img: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        sequence_length = input_ids.shape[1]
        total_len = sequence_length + num_pred_img + num_add_img
        min_dtype = torch.finfo(dtype).min
        indices = (input_ids == 7).nonzero()
        mul = int(indices.shape[0] / input_ids.shape[0])
        # print("log mul", mul)
        attn_masks = []
        boi_pos = []
        for id in range(input_ids.shape[0]):
            # print(input_ids[id])
            causal_mask = torch.full(
                (total_len, total_len), fill_value=min_dtype, dtype=dtype, device=device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            start = indices[id * mul][-1] + 1
            boi_pos.append(start - 1)
            causal_mask[start:start + num_add_img,start:start + num_add_img] = causal_mask[0][0]
            # if num_pred_img > 0:
            #     causal_mask[total_len - num_pred_img:total_len, total_len - num_pred_img:total_len] = causal_mask[0,0] # bidirectional attention
            # print(start)
            causal_mask[:, :start-2] = min_dtype
            attn_masks.append(torch.unsqueeze(causal_mask, 0))
        attn_masks = torch.stack(attn_masks)
        return attn_masks, boi_pos

    def prepare_causal_attention_mask_for_text_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_add_img: int,
        dtype: torch.dtype,
        device: torch.device,
        max_pred_tokens: int
    ):
        sequence_length = input_ids.shape[1]
        sequence_length = sequence_length + num_add_img
        target_length = sequence_length + max_pred_tokens
        min_dtype = torch.finfo(dtype).min
        indices = (input_ids == 7).nonzero()
        mul = int(indices.shape[0] / input_ids.shape[0])
        attn_masks = []
        boi_pos = []
        for id in range(input_ids.shape[0]):
            # print(input_ids[id])
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            start = indices[id * mul][-1] + 1
            boi_pos.append(start - 1)
            causal_mask[start:start + num_add_img,start:start + num_add_img] = causal_mask[0][0] # bidirectional for image tokens
            causal_mask[:, :start-2] = min_dtype # pad tokens
            attn_masks.append(torch.unsqueeze(causal_mask, 0))
        attn_masks = torch.stack(attn_masks)
        return attn_masks, boi_pos

    def get_last_layer(self):
        return self.mlp_llm_2_image.fc2.weight
    
    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        # print("=========== log in calculate_adaptive_weight", torch.norm(recon_grads), torch.norm(g_grads) + 1e-4, d_weight)
        return d_weight
    
    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight
    
    def forward_t2i(self, input_ids, attention_masks, img_llm_inputs):
        text_emb = self.llm.model.get_input_embeddings()(input_ids)
        # print("text_emb", text_emb.shape)
        attention_masks_4d, boi_pos = self.prepare_causal_attention_mask_for_visual(
            input_ids=input_ids,
            attention_mask=attention_masks,
            num_add_img=self.num_patches,
            num_pred_img=self.num_patches,
            device=text_emb.device,
            dtype=text_emb.dtype
        )
        placeholder_token = self.placeholder_token.repeat(1, self.num_patches, 1)
        # placeholder_token = placeholder_token + self.decoder_pos_embed
        # placeholder_token = placeholder_token + self.decoder_pos_embed
        # print("placeholder_token", placeholder_token.shape)
        placeholder_token = placeholder_token.repeat(input_ids.shape[0], 1, 1)
        # print("placeholder_token", placeholder_token.shape)
        # print("img_llm_inputs", img_llm_inputs.shape)
        input_embs = []
        for id in range(img_llm_inputs.shape[0]):
            boi_id = boi_pos[id]
            # tmp1 = text_emb[id, :boi_id+1,:]
            # print("tmp1", tmp1.shape)
            input_emb = torch.cat((text_emb[id, :boi_id+1,:], img_llm_inputs[id], text_emb[id, boi_id+1:,:], placeholder_token[id]), dim=0)
            # print("input_emb", input_emb.shape)
            input_embs.append(input_emb)
        input_embs = torch.stack(input_embs)
        # print(input_embs.shape)
        # print(attention_masks_4d.shape)
        # print("before t2i", input_embs.shape)
        # print("before t2i", attention_masks_4d.shape)
        llm_output = self.llm({
            "inputs_embeds": input_embs,
            "attention_mask": attention_masks_4d,
            # "feature_only": True
        })
        # for key in llm_output:
        #     print(key, llm_output[key].shape)
        # return llm_output
        return llm_output["last_hidden_state"][:, -self.num_patches:, :]

    def forward_i2t(self, input_ids, attention_masks, img_llm_inputs, labels):
        text_emb = self.llm.model.get_input_embeddings()(input_ids)
        attention_masks_4d, boi_pos = self.prepare_causal_attention_mask_for_visual(
            input_ids=input_ids,
            attention_mask=attention_masks,
            num_add_img=self.num_patches,
            num_pred_img=0,
            device=text_emb.device,
            dtype=text_emb.dtype
        )
        input_embs = []
        new_labels = []
        for id in range(img_llm_inputs.shape[0]):
            boi_id = boi_pos[id]
            l = labels[id]
            l = torch.concat([torch.from_numpy(np.array([-100] * self.num_patches)).to(l.dtype).to(l.device), l])
            new_labels.append(l)
            # tmp1 = text_emb[id, :boi_id+1,:]
            # print("tmp1", tmp1.shape)
            input_emb = torch.cat((text_emb[id, :boi_id+1,:], img_llm_inputs[id], text_emb[id, boi_id+1:,:]), dim=0)
            # print("input_emb", input_emb.shape)
            input_embs.append(input_emb)
        # tmp = labels.one_li
        
        # labels = torch.cat((labels[:,0].repeat(1, 1024), labels), dim=0)
        input_embs = torch.stack(input_embs)
        new_labels = torch.stack(new_labels)
        # print("before i2t", attention_masks_4d.shape)
        # print("before i2t", input_embs.shape)
        # print("before i2t", labels.shape)
        llm_output = self.llm({
            "inputs_embeds": input_embs,
            "attention_mask": attention_masks_4d,
            "labels": new_labels
        })
        return llm_output
    
    def forward_loss_t2i(self, pred_img=None, gt_img=None, current_step = -1):
        """
        pred_latent: [N, 1024]
        gt_latent: [N, 1024]
        """
        # print("---------------------- log", pred_img.dtype)
        ret = {}
        ret["total_pix_loss"] = 0
        if "pixel_opt" in self.cfgs["loss"]:
            ret["pixel_loss"] = self.pix_loss(pred_img, gt_img) 
            ret["total_pix_loss"] += ret["pixel_loss"] * self.cfgs["loss"]["pixel_opt"]["loss_weight"]
        if "perceptual_opt" in self.cfgs["loss"]:
            ret["percept_loss"] = self.perceptual_loss(pred_img, gt_img)
            ret["total_pix_loss"] += ret["percept_loss"] * self.cfgs["loss"]["perceptual_opt"]["loss_weight"]
        if "gan_opt" in self.cfgs["loss"]:
            if self.training:
                net_d_pred_fake_detach = self.net_d(pred_img.detach())
                net_d_pred_real = self.net_d(gt_img)
                l_d_real = self.ganloss(net_d_pred_real, True, is_disc=True)
                l_d_fake = self.ganloss(net_d_pred_fake_detach, False, is_disc=True)
                ret['l_d_real'] = l_d_real
                ret['out_d_real'] = torch.mean(net_d_pred_real.detach())
                ret['l_d_fake'] = l_d_fake
                ret['out_d_fake'] = torch.mean(net_d_pred_fake_detach.detach())
                ret["disc_loss"] = l_d_real + l_d_fake
                if current_step > self.cfgs["loss"]["gan_opt"]["net_g_start_train"]:
                    net_d_pred_fake = self.net_d(pred_img)
                    l_g_gan = self.ganloss(net_d_pred_fake, True, is_disc=False)
                    last_layer = self.get_last_layer()
                    d_weight = self.calculate_adaptive_weight(ret["total_pix_loss"], l_g_gan, last_layer, disc_weight_max=1.0)
                    d_weight *= self.cfgs["loss"]["gan_opt"]["disc_weight"]
                    ret["total_pix_loss"] += d_weight * l_g_gan
                    ret['l_g_fake'] = l_g_gan
                    ret['g_weight'] = d_weight


        return ret
    
    def text_generate(self, input_ids, attention_masks, pixel_values, max_gen_token=128):
        with torch.no_grad():
            # bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.mean
            # bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
            bad_img_tokens = self.tokenize_image(pixel_values=pixel_values)
            bad_img_tokens, h_, w_ = self.patchify(bad_img_tokens, self.patch_group)
            img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
            # img_llm_inputs += self.decoder_pos_embed
            text_emb = self.llm.model.get_input_embeddings()(input_ids)
            attention_masks_4d, boi_pos = self.prepare_causal_attention_mask_for_text_generation(
                input_ids=input_ids,
                attention_mask=attention_masks,
                num_add_img=self.num_patches,
                device=text_emb.device,
                dtype=text_emb.dtype,
                max_pred_tokens=max_gen_token
            )
            # torch.ones(size=)
            attention_masks = torch.cat(
                (attention_masks, torch.ones(size=(attention_masks.shape[0], self.num_patches), dtype=attention_masks.dtype, device=attention_masks.device)),
                dim=1
            )
            input_embs = []
            for id in range(img_llm_inputs.shape[0]):
                boi_id = boi_pos[id]
                input_emb = torch.cat((text_emb[id, :boi_id+1,:], img_llm_inputs[id], text_emb[id, boi_id+1:,:]), dim=0)
                input_embs.append(input_emb)
            input_embs = torch.stack(input_embs)
            input_dict = {}
            input_dict["inputs_embeds"] = input_embs
            input_dict["attention_mask"] = attention_masks
            input_dict["kwargs"] = {"attention_mask_kickoff": attention_masks_4d}
            output = self.llm.model.generate(**input_dict, max_new_tokens = max_gen_token)
            return output
    
    def image_generate(self, pixel_values=None, input_ids_t2i=None, attention_masks_t2i=None):
        with torch.no_grad():
            # bad_img_tokens = self.img_vae.encode(pixel_values).latent_dist.mean
            # bad_img_tokens = bad_img_tokens.to(self.mlp_image_2_llm.fc1.weight.dtype)
            bad_img_tokens = self.tokenize_image(pixel_values=pixel_values)
            bad_img_tokens, h_, w_ = self.patchify(bad_img_tokens, self.patch_group)
            img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
            # img_llm_inputs += self.decoder_pos_embed
            llm_output_t2i = self.forward_t2i(
                input_ids=input_ids_t2i,
                attention_masks=attention_masks_t2i,
                img_llm_inputs=img_llm_inputs
            )
            img_token_pred = self.mlp_llm_2_image(llm_output_t2i)
            img_latent = self.unpatchify(img_token_pred, p=self.patch_group, dim=self.vae_dim, h_=h_, w_=w_)
            img_pred = self.detokenize_image(img_latent=img_latent)
            # img_latent = img_latent.to(self.img_vae.dtype)
            # img_pred = self.img_vae.decode(img_latent).sample # img_pred in range [-1, 1]
        return img_pred
        
    def forward(self, pixel_values=None, target_image=None, input_ids_t2i=None, attention_masks_t2i=None, input_ids_i2t=None, attention_masks_i2t=None, labels=None, gen_text_only = False, gen_image_only = False, current_iter=None):
        if gen_image_only == True:
            return self.image_generate(
                pixel_values=pixel_values,
                input_ids_t2i=input_ids_t2i,
                attention_masks_t2i=attention_masks_t2i
            )
        if gen_text_only == True:
            return self.text_generate(
                pixel_values=pixel_values,
                input_ids=input_ids_t2i,
                attention_masks=attention_masks_t2i
            )
        bad_img_tokens = self.tokenize_image(pixel_values=pixel_values)
        bad_img_tokens, h_, w_ = self.patchify(bad_img_tokens, self.patch_group)
        img_llm_inputs = self.mlp_image_2_llm(bad_img_tokens)
        # img_llm_inputs += self.decoder_pos_embed
        # print("============================== debug grad_fn llm_inputs", llm_inputs.grad_fn)
        llm_output_t2i = self.forward_t2i(
            input_ids=input_ids_t2i,
            attention_masks=attention_masks_t2i,
            img_llm_inputs=img_llm_inputs
        )
        img_token_pred = self.mlp_llm_2_image(llm_output_t2i)
        img_latent = self.unpatchify(img_token_pred, p=self.patch_group, dim=self.vae_dim, h_=h_, w_=w_)
        img_pred = self.detokenize_image(img_latent=img_latent)
        img_pred = torch.clamp(img_pred, -1.0, 1.0)
        loss_dict = self.forward_loss_t2i(pred_img=img_pred, gt_img=target_image, current_step=current_iter)
        loss_dict["pred_imgs"] = img_pred 
        if input_ids_i2t is not None:
            llm_output_i2t = self.forward_i2t(
                input_ids=input_ids_i2t,
                attention_masks=attention_masks_i2t,
                img_llm_inputs=img_llm_inputs,
                labels=labels
            )
            loss_dict["total_text_loss"] = llm_output_i2t["loss"] 
        # print("img_pred", img_pred.shape)
        # print()
        # print()
        # print(llm_output_t2i)
        # print("llm_output_i2t")
        # print(llm_output_i2t)
        return loss_dict


        # # print("log llm output", img_token_pred.shape)
        # # print("============================== debug grad_fn img_token_pred", img_token_pred.grad_fn)
        
        # # print("log llm output", img_token_pred.shape)
        # # print("============================== debug grad_fn img_token_pred after unpatchify", img_token_pred.grad_fn)
        
        # # img_latent = img_latent.to(self.img_decoder._dtype)
        
        # # img_pred = self.img_decoder.decode(img_latent) # img_pred in range [-1, 1]
        # # print("============================== debug grad_fn img_token_pred before decode", img_token_pred.grad_fn)
        
        # # img_pred = self.img_vae.decode(img_latent).sample
        # # print("============================== debug grad_fn img_pred", img_pred.grad_fn)
        # # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_using"]:
    
        # # if "gan_opt" in self.cfgs["loss"] and current_iter >= self.cfgs["loss"]["gan_opt"]["net_d_start_train"] and gt_images is not None:
        
        # return {
        #     "pred_token": img_token_pred,
        #     "pred_img": img_pred,
        #     "gt_img": gt_images
        # }


if __name__ == "__main__":
    with open("/scratch2/f0072r1/res_gemma/cfgs/mae_gemma_denoise.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = ResLLM(cfgs=None)
    print(model)
    # exit(0)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # image_processor = AutoImageProcessor.from_pretrained("hf_w/vit-mae-large", use_fast=True)

    # inputs = image_processor(images=image, return_tensors="pt")
    # output = model(inputs)

    # print(output)
    # print(output.shape)

    


