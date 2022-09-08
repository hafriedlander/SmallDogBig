
import os
import cv2
import torch
from torch.nn import functional as F
import numpy as np
import gdown

from basicsr.utils.download_util import download_file_from_google_drive
from basicsr.utils.img_util import tensor2img
from models.hat_arch import HAT

def cache_from_gdrive(path, url):
    if not os.path.exists(path):
        gdown.download(url=url, output=path, quiet=False, fuzzy=True)

    return path

def HAT_LargeSR(scale, tile):
    if scale != 2 and scale != 4:
        assert("Only scale 2 or 4 is supported")

    # Taken directly from HAT-L_SRx4_ImageNet-pretrain.yml
    model=HAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )

    if scale==2:
        model_path=cache_from_gdrive('weights/HAT/HAT-L_SRx2_ImageNet-pretrain.pth', 'https://drive.google.com/file/d/16xtMezHvckdWEuSiOxcO-dgOlsI0rEUg/view?usp=sharing')
    else:
        model_path=cache_from_gdrive('weights/HAT/HAT-L_SRx4_ImageNet-pretrain.pth', 'https://drive.google.com/file/d/1vUiknHsRuqhZN25dt2y3jnoPu5SWjRkP/view?usp=sharing')

    return HATHelper(
        scale=scale,
        model=model,
        model_path=model_path,
        tile=tile
    )

def HAT_MidSR(scale, tile):
    if scale != 2 and scale != 4:
        assert("Only scale 2 or 4 is supported")

    # Taken directly from HAT_SRx4_ImageNet-pretrain.yml
    model=HAT(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv'
    )    
    if scale==2:
        model_path=cache_from_gdrive('weights/HAT/HAT_SRx2_ImageNet-pretrain.pth', 'https://drive.google.com/file/d/11WDyK4MMcRapHs_aKJKHaYAFsb29SoCw/view?usp=sharing')
    else:
        model_path=cache_from_gdrive('weights/HAT/HAT_SRx4_ImageNet-pretrain.pth', 'https://drive.google.com/file/d/1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI/view?usp=sharing')

    return HATHelper(
        scale=scale,
        model=model,
        model_path=model_path,
        tile=tile
    )

class HATHelper():
    """A helper class for upsampling images with HAT.

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): As too large images result in the out of GPU memory issue, so this tile option will first crop
            input images into tiles, and then process each of them. Finally, they will be merged into one image.
            0 denotes for do not use tile. Default: 0.
        tile_pad (int): The pad size for each tile, to remove border artifacts. Default: 10.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        half (float): Whether to use half precision during inference. Default: False.
    """

    def __init__(self,
                 scale,
                 model_path,
                 model,
                 tile=0,
                 tile_overlap=32,
                 device=None
                ):

        self.scale = scale
        self.window_size = model.window_size
        self.tile = tile
        self.tile_overlap = tile_overlap

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['params_ema'] if 'params_ema' in pretrained_model.keys() else pretrained_model, strict=True)
        model.eval()
        self.model = model.to(self.device)

    def upsample(self, img):
        """
        Img is expected to be an RGB image that is a multiple of window_size.
        This function will run the actual upsample, either in one pass, or tiled, based on self.tile
        """

        if self.tile == 0:
            # test the image as a whole
            output = self.model(img)
        
        else:
            # test the image tile by tile
            b, c, h, w = img.size()
            tile = min(self.tile, h, w)
            assert tile % self.window_size == 0, "tile size should be a multiple of window_size"
            tile_overlap = self.tile_overlap
            sf = self.scale

            stride = tile - tile_overlap
            h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
            w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
            E = torch.zeros(b, c, h*sf, w*sf).type_as(img)
            W = torch.zeros_like(E)

            for h_idx in h_idx_list:
                for w_idx in w_idx_list:
                    in_patch = img[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                    out_patch = self.model(in_patch)
                    out_patch_mask = torch.ones_like(out_patch)

                    E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                    W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)

            output = E.div_(W)

        return output

    @torch.no_grad()
    def pad_and_upsample(self, img):
        """
        Img is expected to be an RGB image, but doesn't need to be a multiple of window_size
        This function will pad the image to be a multiple, run the upsample, and then strip the pad back off
        """
        # pad input image to be a multiple of window_size
        window_size = self.window_size
        scale = self.scale
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = img.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), 'reflect')

        output_img_t=self.upsample(img)

        _, _, h, w = output_img_t.size()
        output_img_t=output_img_t[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

        result = output_img_t.detach().cpu()
        output_img = tensor2img(result, out_type=np.float)

        #output_img = output_img_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        #output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        return output_img

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='hat'):
        h_input, w_input = img.shape[0:2]

        # img: numpy
        img = img.astype(np.float32)
        if np.max(img) > 256:  # 16-bit image
            max_range = 65535
            print('\tInput is a 16-bit image')
        else:
            max_range = 255
        img = img / max_range

        if len(img.shape) == 2:  # gray image
            img_mode = 'L'
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA image with alpha channel
            img_mode = 'RGBA'
            alpha = img[:, :, 3]
            img = img[:, :, 0:3]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha_upsampler == 'hat':
                alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2RGB)
        else:
            img_mode = 'RGB'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        output_img = self.pad_and_upsample(img)
        if img_mode == 'L': output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)

        torch.cuda.empty_cache()        

        # ------------------- process the alpha channel if necessary ------------------- #
        if img_mode == 'RGBA':
            if alpha_upsampler == 'hat':
                output_alpha = self.pad_and_upsample(alpha)
                output_alpha = cv2.cvtColor(output_alpha, cv2.COLOR_BGR2GRAY)
            else:  # use the cv2 resize for alpha channel
                h, w = alpha.shape[0:2]
                output_alpha = cv2.resize(alpha, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR)

            # merge the alpha channel
            output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2BGRA)
            output_img[:, :, 3] = output_alpha

        # ------------------------------ return ------------------------------ #

        if max_range == 65535:  # 16-bit image
            output = (output_img * 65535.0).round().astype(np.uint16)
        else:
            output = (output_img * 255.0).round().astype(np.uint8)

        if outscale is not None and outscale != float(self.scale):
            output = cv2.resize(
                output, (
                    int(w_input * outscale),
                    int(h_input * outscale),
                ), interpolation=cv2.INTER_LANCZOS4)

        return output, img_mode
