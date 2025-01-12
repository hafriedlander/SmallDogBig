## SmallDogBig

This is a quick CLI utility I threw together from other people's work (see credits below)
to upsample Stable Diffusion outputs (although it should work nicely on other things too)

It optionally fixes faces, and uses deep learning to scale up all the images in a directory

### Install / Use

It _should_ be enough to put some input images into the `inputs` folder and then do

```
conda env create -f environment.yaml
conda activate smalldogbig
```

And then

```
python smalldogbig.py 
```

You can specify the upsampler by doing --bg_upsampler {option}. 
Can be one of: None, swinir (default, x4), swinir_x2, realesrgan (x4), realesrgan_x2, realesrgan_anime, hat (x4), hat_x2, edt
- realesrgan_anime is fastest, gives nice results on anime and illustrations, removes fine detail
- realesrgan is middle
- swinir is much slower, gives the most details
- hat gives the best results on real photos, but is often worse than realesrgan or swinir on synthetic images
- edt is an alternative to hat - very similar (although not quite as good) on real photos, but much faster
- The x2 versions scale to x2 instead of x4 before processing. They're faster, but otherwise worse

You will need at least 6GB of VRAM most upscalers. You can try setting a smaller bg_tile for 4GB.

You can adjust face correction with --w {adjust} which is 0-1. 
- 0 gives more correction at the expense of accuracy
- 1 tries to be more accurate at the expense of correction.

Use 0.7 - 0.9 for mostly good looking faces, or 0.2 for messed up faces.

### Weights

All of the scalers will auto-download their weights, _except_ EDT.

For EDT, download all the SRx4_EDTB_* weights from https://mycuhk-my.sharepoint.com/:f:/g/personal/1155137927_link_cuhk_edu_hk/Eikt_wPDrIFCpVpiU0zYNu0BwOhQIHgNWuH1FYZbxZhq_w?e=bVEVeW and put them in weights\EDT

### Examples

Original, linear interpolated 4x for comparison
![Original, linear interpolated to 4x](/outputs/examples/PrincessSummerFruit_nofaces.png?raw=true "Original, linear interpolated to 4x")

SwinIR + Strong Face Correction
![SwinIR + Strong Face Correction](/outputs/examples/PrincessSummerFruit_bgsr_swinir_facesr_swinir.png?raw=true "SwinIR + Strong Face Correction")

[Examples of other modes here](/outputs/examples/)

### Credits

This is primarily just [CodeFormer](https://github.com/sczhou/CodeFormer) altered to
use [SwinIR](https://github.com/JingyunLiang/SwinIR) as the upscaler, and the CLI tool
simplified for my use case.

### License

This is derived from CoDeformer, which is CC-BY-NC-SA/4.0. This is therefore also CC-BY-NC-SA/4.0.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

Component licenses:

- Codeformer CC-BY-NC-SA/4.0
- BasicSR: Apache 2.0
- SwinIR: Apache 2.0
- HAT: MIT
- EDT: None listed

City.png example image is derived (a scaled-down crop) from https://unsplash.com/photos/wpU4veNGnHg
