import numpy as np
from PIL import Image


"""
    input_mode : 
    1 (1-bit pixels, black and white, stored with one pixel per byte)
    L (8-bit pixels, black and white)
    P (8-bit pixels, mapped to any other mode using a color palette)
    RGB (3x8-bit pixels, true color)
    RGBA (4x8-bit pixels, true color with transparency mask)
    CMYK (4x8-bit pixels, color separation)
    YCbCr (3x8-bit pixels, color video format)
        Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
    LAB (3x8-bit pixels, the L*a*b color space)
    HSV (3x8-bit pixels, Hue, Saturation, Value color space)
    I (32-bit signed integer pixels)
    F (32-bit floating point pixels)
"""
def resize_image_array_set(image_sets, w_in, h_in, input_mode="RGB", resize=False, w_resize=200, h_resize=200,
                           channel_out=1):
    new_shape = (image_sets.shape[0], h_resize, w_resize, channel_out)
    output = np.empty(new_shape)
    for index, image in enumerate(image_sets):
        resize_image = resize_image_array(image, w_in, h_in, input_mode, resize, w_resize, h_resize, channel_out)
        output[index] = resize_image
    return output


def resize_image_array(image_array, w_in, h_in, input_mode="RGB", resize=False, w_resize=200, h_resize=200,
                       channel_out=1):
    image = image_encode(image_array, W=w_in, H=h_in, MODE=input_mode)
    if resize:
        img_resize = image.resize((w_resize, h_resize), Image.LANCZOS)
        # img_resize.show()
        image = img_resize

    output_array = image_decode(image, input_mode="RGB", channel_out=channel_out)

    return output_array


def image_encode(image_array, W=32, H=32, MODE="RGB"):
    image_reshape = image_array
    if MODE is "RGB":
        if np.ndim(image_array) == 1:
            image_reshape = np.reshape(image_array, newshape=[3, W, H]).transpose([1, 2, 0])

    image = Image.fromarray(image_reshape, mode=MODE)
    return image


def image_decode(image, input_mode="RGB", channel_out=1):
    image_array = np.asarray(image)
    # if input_mode is "RGB":
    #     image_array = image_array.transpose([2, 0, 1])

    image_out = image_array
    return image_out
