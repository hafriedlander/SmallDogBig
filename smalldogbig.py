# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py
import os
from time import time
import cv2
import argparse
import glob
import warnings
import torch
import pynvml

from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
import archs

# Tile defaults for video memory in GB
bg_tile_defaults = {
    'hat': ((0, 64), (7, 112), (11, 192)),
    'edt': ((0, 96), (7, 144), (11, 384)),
    'swi': ((0, 128), (7, 192), (11, 512)),
    'def': ((0, 256), (7, 384), (11, 768))
}

# Default to assuming 6GB VRAM
vram_total_gb=0

try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    vram_total_gb = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024 * 1024)
except:
    print("Unable to determine available VRAM. Using small bg_tile. You may be able to use larger bg_tile values for higher speed.")


pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_swinir():
    from scalers.SwinIRHelper import SwinIR_LargeSR
    return SwinIR_LargeSR(4, args.bg_tile)

def set_swinir_x2():
    from scalers.SwinIRHelper import SwinIR_MidSR
    return SwinIR_MidSR(2, args.bg_tile)

def set_hat():
    from scalers.HATHelper import HAT_LargeSR
    return HAT_LargeSR(4, args.bg_tile)

def set_hat_x2():
    from scalers.HATHelper import HAT_LargeSR
    return HAT_LargeSR(2, args.bg_tile)

def set_realesrgan():
    from scalers.RealESRGANHelper import RealESRGAN_x4
    return RealESRGAN_x4(args.bg_tile)

def set_realesrgan_anime():
    from scalers.RealESRGANHelper import RealESRGAN_Animex4
    return RealESRGAN_Animex4(args.bg_tile)

def set_realesrgan_x2():
    from scalers.RealESRGANHelper import RealESRGAN_x2
    return RealESRGAN_x2(args.bg_tile)

def set_edt():
    from scalers.EDTHelper import EDT_SR
    return EDT_SR(4, args.bg_tile)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--w', type=float, default=0.7, help='Balance the quality and fidelity')
    parser.add_argument('--upscale', type=int, default=4, help='The final upsampling scale of the image. Default: 4')
    parser.add_argument('--in_path', type=str, default='./inputs')
    parser.add_argument('--out_path', type=str, default='./outputs')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50')
    parser.add_argument('--draw_box', action='store_true')
    parser.add_argument('--bg_upsampler', type=str, default='swinir', help='Background upsampler. None, realesrgan, realesrgan_x2, realesrgan_anime, swinir, swinir_x2, hat, hat_x2')
    parser.add_argument('--no_face_upsample', action='store_true', help='Disable face upsampler after enhancement (if bg_upsampler is not None)')
    parser.add_argument('--bg_tile', type=int, default=-1, help=f'Tile size for background sampler. Default: depends on upsampler')
    parser.add_argument('--no_face_correction', action='store_true', help='Disable face correction (just do upscaling)')
    parser.add_argument('--save_intermediates', action='store_true', help='Also save just the detected faces, original and restored, for analysis')

    args = parser.parse_args()

    # Calculate default bg_tile for vram & upsampler model
    if args.bg_tile == -1:
        defaults = bg_tile_defaults.get(args.bg_upsampler[:3], bg_tile_defaults['def'])
        *_, args.bg_tile = (x[1] for x in defaults if x[0] <= vram_total_gb)

    # ------------------------ input & output ------------------------
    if args.in_path.endswith('/'):  # solve when path ends with /
        args.in_path = args.in_path[:-1]

    result_root = os.path.join(args.out_path, os.path.basename(args.in_path))

    # ------------------ set up background upsampler ------------------
    bg_upsampler = None

    if args.bg_upsampler:
        if not torch.cuda.is_available():  # CPU
            warnings.warn(
                'Currently we prevent using upsamplers if you don\'t have CUDA.'
                'If you really want to use it, please modify the corresponding codes.',
                category=RuntimeWarning
            )
        
        elif args.bg_upsampler == 'realesrgan':
            bg_upsampler = set_realesrgan()
        elif args.bg_upsampler == 'realesrgan_x2':
            bg_upsampler = set_realesrgan_x2()
        elif args.bg_upsampler == 'realesrgan_anime':
            bg_upsampler = set_realesrgan_anime()
        elif args.bg_upsampler == 'swinir':
            bg_upsampler = set_swinir()
        elif args.bg_upsampler == 'swinir_x2':
            bg_upsampler = set_swinir_x2()
        elif args.bg_upsampler == 'hat':
            bg_upsampler = set_hat()
        elif args.bg_upsampler == 'hat_x2':
            bg_upsampler = set_hat_x2()
        elif args.bg_upsampler == 'edt':
            bg_upsampler = set_edt()
        elif args.bg_upsampler != "None" and args.bg_upsampler != "none":
            warnings.warn(
                f'Unknown upsampler {args.bg_upsampler} requested. Nothing will be used',
                category=RuntimeWarning
            )

    # ------------------ set up face upsampler ------------------
    face_upsampler = None
    if not args.no_face_upsample: face_upsampler = bg_upsampler

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    # large det_model: 'YOLOv5l', 'retinaface_resnet50'
    # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
    if args.no_face_correction:
        print("No face correction")
    elif not args.has_aligned: 
        print(f'Face detection model: {args.detection_model}')
    
    print(f'Background upsampling: {bg_upsampler is not None}, Face upsampling: {face_upsampler is not None}')

    face_helper = FaceRestoreHelper(
        args.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model = args.detection_model,
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    # scan all the jpg and png images
    for img_path in sorted(glob.glob(os.path.join(args.in_path, '*.[jp][pn]g'))):
        # clean all the intermediate results to process the next image
        face_helper.clean_all()
        
        img_name = os.path.basename(img_path)
        print(f'Processing: {img_name}')
        basename, ext = os.path.splitext(img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # paste_back
        if not args.has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                bg_img = bg_upsampler.enhance(img, outscale=args.upscale)[0]
            else:
                h, w = img.shape[0:2]
                bg_img = cv2.resize(img, (w * args.upscale, h * args.upscale), interpolation=cv2.INTER_LINEAR)
        
        if args.no_face_correction:
            restored_img = bg_img

        else:
            if args.has_aligned: 
                # the input faces are already cropped and aligned
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                # get face landmarks for each face
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                # align and warp each face
                face_helper.align_warp_face()

            # face restoration for each cropped face
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                # prepare data
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=args.w, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}')
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face)

            face_helper.get_inverse_affine(None)

            # paste each restored face to the input image
            if face_upsampler is not None: 
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box, face_upsampler=face_upsampler)
            else:
                restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=args.draw_box)

            # save faces
            if args.save_intermediates:
                for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
                    # save cropped face
                    if not args.has_aligned: 
                        save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
                        imwrite(cropped_face, save_crop_path)
                    # save restored face
                    if args.has_aligned:
                        save_face_name = f'{basename}.png'
                    else:
                        save_face_name = f'{basename}_{idx:02d}.png'
                    save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
                    imwrite(restored_face, save_restore_path)

        # save restored img
        if not args.has_aligned and restored_img is not None:
            postfix=""
            if bg_upsampler is not None: postfix += "_bgsr_" + args.bg_upsampler
            if face_upsampler is not None: postfix += "_facesr_" + args.bg_upsampler
            if args.no_face_correction: postfix += "_nofaces"

            save_restore_path = os.path.join(result_root, f'{basename}{postfix}.png')
            imwrite(restored_img, save_restore_path)

    print(f'\nAll results are saved in {result_root}')
