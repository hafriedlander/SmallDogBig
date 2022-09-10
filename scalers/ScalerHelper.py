import math
import cv2
import torch
import numpy as np
from torch.nn import functional as F

class ScalerHelper():
    """A helper class for upsampling images

    Args:
        scale (int): Upsampling scale factor used in the networks. It is usually 2 or 4.
        model_path (str): The path to the pretrained model. It can be urls (will first download it automatically).
        model (nn.Module): The defined network. Default: None.
        tile (int): The total size including overlap of a tile. 0 means don't tile. Default: 0
        tile_overlap (float): How much each tile should overlap it's neighbour tiles (and be discarded after processing). Default 0.25.
        pre_pad (int): Pad the input images to avoid border artifacts. Default: 10.
        window_size (int): The window size of the model
    """

    def __init__(self,
                 scale,
                 model_path,
                 model,
                 tile=0,
                 tile_overlap=0.25,
                 pre_pad=32,
                 window_size=0,
                 device=None
                ):

        self.scale = scale
        self.tile = tile
        self.tile_overlap = math.ceil(tile*tile_overlap)
        self.pre_pad = pre_pad
        self.window_size = window_size

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['params_ema'] if 'params_ema' in pretrained_model.keys() else pretrained_model, strict=True)
        model.eval()
        self.model = model.to(self.device)

    def process(self, img):
        # model inference
        return self.model(img)

    def tile_process(self, img):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        pad=self.tile_overlap
        stride=self.tile-pad*2

        # start with black image
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / stride)
        tiles_y = math.ceil(height / stride)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * stride
                ofs_y = y * stride
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + stride, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + stride, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - pad, 0)
                input_end_x_pad = min(input_end_x + pad, width)
                input_start_y_pad = max(input_start_y - pad, 0)
                input_end_y_pad = min(input_end_y + pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = self.process(input_tile)
                except RuntimeError as error:
                    print('Error', error)

                # output tile area on total image
                output_start_x = input_start_x * self.scale
                output_end_x = input_end_x * self.scale
                output_start_y = input_start_y * self.scale
                output_end_y = input_end_y * self.scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
        return output

    def upsample(self, img):
        """
        Img is expected to be an RGB image that is a multiple of window_size.
        This function will run the actual upsample, either in one pass, or tiled, based on self.tile
        """

        if self.tile == 0:
            return self.process(img)
        else:
            return self.tile_process(img)

    @torch.no_grad()
    def pad_and_upsample(self, img):
        """
        Img is expected to be an RGB image, but doesn't need to be a multiple of window_size
        This function will pad the image to be a multiple, run the upsample, and then strip the pad back off
        """

        # pad input image to be a multiple of window_size, but with a minimum pad of pre_pad

        _, _, h_old, w_old = img.size()
        h_pad = math.ceil((h_old+self.pre_pad*2) / self.window_size) * self.window_size - h_old
        w_pad = math.ceil((w_old+self.pre_pad*2) / self.window_size) * self.window_size - w_old

        l_pad=w_pad//2
        r_pad=w_pad-l_pad
        t_pad=h_pad//2
        b_pad=h_pad-t_pad

        # Add padding
        img = F.pad(img, (l_pad, r_pad, t_pad, b_pad), 'reflect')
        # Upsample
        output_img_t = self.upsample(img)
        # Remove padding
        output_img_t = output_img_t[..., l_pad*self.scale:(l_pad+h_old) * self.scale, t_pad*self.scale:(t_pad+w_old) * self.scale]

        # Convert back from tensor to cv2 Numpy image
        output_img = output_img_t.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))

        return output_img

    @torch.no_grad()
    def enhance(self, img, outscale=None, alpha_sr=True):
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
            if alpha_sr:
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
            if alpha_sr:
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
