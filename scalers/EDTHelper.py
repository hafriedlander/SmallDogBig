from models.edt import Network
from scalers.ScalerHelper import ScalerHelper

def lcm(ab):
    a, b = ab[0], ab[1]
    for i in range(min(a, b), 0, -1):
        if a % i == 0 and b % i == 0:
            return a * b // i

def EDT_SR(scale, tile):
    if scale != 4:
        assert("Only scale 4 is supported")

    from scalers.EDTconfigs.SRx4_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K import Config
    model = Network(Config)

    return EDTHelper(
        scale=scale,
        model=model,
        model_path='weights/EDT/SRx4_EDTB_Div2kFlickr2K__SRx2x3x4_ImageNet200K.pth',
        tile=tile,
        window_size=lcm(Config.MODEL.WINDOW_SIZE)
    )

class EDTHelper(ScalerHelper):
    def process(self, img):
        return self.model(img)[0]
