import os
import cv2
import torch
import numpy as np
from basicsr.utils.download_util import load_file_from_url
from models.network_swinir import SwinIR

def SwinIR_MidSR(scale, tile):
    if scale != 2 and scale != 4:
        assert("Only scale 2 or 4 is supported")

    model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

    return SwinIRHelper(
        scale=scale,
        model=model,
        model_path='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x{scale}_GAN.pth',
        tile=tile
    )

def SwinIR_LargeSR(scale, tile):
    if scale != 4:
        assert("Only scale 4 is supported")

    model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')

    return SwinIRHelper(
        scale=4,
        model=model,
        model_path='https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
        tile=tile
    )

class SwinIRHelper():
    """A helper class for upsampling images with SwinIRer.

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

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join('weights/SwinIR'), progress=True, file_name=None)

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
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
        w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
        output_img_t = self.upsample(img)
        output_img_t = output_img_t[..., :h_old * self.scale, :w_old * self.scale]

        output_img = output_img_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        return output_img

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_upsampler='swinir'):
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
            if alpha_upsampler == 'swinir':
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
            if alpha_upsampler == 'swinir':
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
