
import os
import gdown

from models.hat_arch import HAT
from scalers.ScalerHelper import ScalerHelper

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

class HATHelper(ScalerHelper):
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
                 tile_overlap=0.25,
                 pre_pad=32,
                 device=None
                ):

        ScalerHelper.__init__(
            self, 
            scale=scale, 
            model_path=model_path, 
            model=model, 
            tile=tile, 
            tile_overlap=tile_overlap, 
            pre_pad=pre_pad, 
            window_size=model.window_size, 
            device=device
        )
