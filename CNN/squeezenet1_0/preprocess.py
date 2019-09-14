import imgaug as ia
from imgaug import augmenters as iaa

def process_image_iaa(img):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes(iaa.Crop(percent=(0, 0.2))),
        sometimes(iaa.ContrastNormalization((0.5, 1.5))),
        sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2))),
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},# scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.SomeOf((0, 4), [
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
        ]),
        sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
            iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
            iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
        ])),
    ], random_order=True)

    image_aug = seq.augment_image(img)
    return image_aug

def process_image_visit(vst):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255), per_channel=0.5)),
        sometimes(iaa.ContrastNormalization((0.5, 1.5))),  ########### we add to equalization
        sometimes(iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2))),
        sometimes(iaa.Add((-5, 5), per_channel=0.5)),  # change brightness of images (by -10 to 10 of original value)
        sometimes(iaa.OneOf([
            iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 3.0
            # iaa.AverageBlur(k=(2, 5)),  # blur image using local means with kernel sizes between 2 and 7
            # iaa.MedianBlur(k=(3, 5)),  # blur image using local medians with kernel sizes between 2 and 7
        ])),
    ], random_order=True)

    visit_aug = seq.augment_image(vst)
    return visit_aug





















