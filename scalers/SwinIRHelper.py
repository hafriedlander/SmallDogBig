import os
from basicsr.utils.download_util import load_file_from_url

from models.network_swinir import SwinIR
from scalers.ScalerHelper import ScalerHelper

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
        model_path=f'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x{scale}_GAN.pth',
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

class SwinIRHelper(ScalerHelper):
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
                 tile_overlap=0.5,
                 pre_pad=32,
                 device=None
                ):

        if model_path.startswith('https://'):
            model_path = load_file_from_url(
                url=model_path, model_dir=os.path.join('weights/SwinIR'), progress=True, file_name=None)

        ScalerHelper.__init__(
            self, 
            scale=scale, 
            model_path=model_path, 
            model=model, 
            tile=tile, 
            tile_overlap=tile_overlap, 
            pre_pad=32, 
            window_size=model.window_size, 
            device=device
        )

