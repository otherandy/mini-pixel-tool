import cv2
import numpy as np
import os
import argparse

# default values
SCALE = 10
BACKGROUND_COLOR = (205, 220, 228)  # BGR
DISPLAY_SIZE = 500
OUTPUT_DIR = './out'

DEBUG = False


def main():
    parser = argparse.ArgumentParser(
        prog='main.py', description='Upscale a mini pixel, convert it to a display image and add a shadow (requires a_Shadow2_MiniPixelDisplay.png in the current directory).',
        epilog='Example: python main.py input.png -s 10 -c e4dccd -d 500 -o \'./out\' --no-shadow --show --no-write')

    parser.add_argument('input_file', type=str,
                        help='path to the input image')
    parser.add_argument('-s', type=int, default=SCALE,
                        dest='scale',
                        required=False,
                        help='scale factor (default: 10)')
    parser.add_argument('-c', type=str, default=bgr_to_hex(BACKGROUND_COLOR),
                        dest='background_color',
                        required=False,
                        help='background color in hex (default: e4dccd)')
    parser.add_argument('-d', type=int, default=DISPLAY_SIZE,
                        dest='display_size',
                        required=False,
                        help='final size of display image in pixels (default: 500)')
    parser.add_argument('-o', type=str, default=OUTPUT_DIR,
                        dest='output_dir',
                        required=False,
                        help='output directory (default: ./out)')

    parser.add_argument('--no-shadow', action='store_true',
                        help='do not add a shadow')
    parser.add_argument('--show', action='store_true',
                        help='show the result, press any key to close the window')
    parser.add_argument('--no-write', action='store_true',
                        help='do not write the result to the output directory')
    args = parser.parse_args()

    input_file = args.input_file
    scale = args.scale
    background_color = hex_to_bgr(args.background_color)
    display_size = args.display_size
    output_dir = args.output_dir
    add_shadow = not args.no_shadow
    write = not args.no_write
    show = args.show

    original = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)

    # scale up the image
    upscaled = cv2.resize(original, (0, 0), fx=scale, fy=scale,
                          interpolation=cv2.INTER_NEAREST)

    # add background
    result = np.full((display_size, display_size, 3),
                     background_color, np.uint8)

    h, w = upscaled.shape[:2]
    hh, ww = result.shape[:2]
    offsetX = (hh - w) // 2
    offsetY = (ww - h) // 2

    if add_shadow:
        add_shadow_image(result, offsetX, offsetY, scale, original.shape[0])
    add_transparent_image(result, upscaled, offsetX, offsetY)

    # result output
    if write:
        write_images(upscaled, result, output_dir)
    if show:
        show_images(original, upscaled, result)


def add_shadow_image(background, offsetX, offsetY, scale, y):
    shadow = cv2.imread(
        'a_Shadow2_MiniPixelDisplay.png', cv2.IMREAD_UNCHANGED)
    if shadow is None:
        print('Error: could not read shadow image a_Shadow2_MiniPixelDisplay.png')
        return

    shadowOffsetX = offsetX - scale
    shadowOffsetY = offsetY + y * scale - scale

    if DEBUG:
        print(f'shadowOffset: ({shadowOffsetX}, {shadowOffsetY})')

    add_transparent_image(background, shadow, shadowOffsetX, shadowOffsetY)


def find_lower_right_pixel(image):
    h, w = image.shape[:2]
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if image[y, x, 3] > 0:
                return x, y
    return None, None


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * \
        (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def show_images(image, upscaled, result):
    cv2.imshow('original', image)
    cv2.imshow('upscaled', upscaled)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_images(upscaled, result, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(output_dir, 'upscaled.png'), upscaled)
    cv2.imwrite(os.path.join(output_dir, 'result.png'), result)


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


def rgb_to_bgr(rgb):
    return rgb[::-1]


def bgr_to_rgb(bgr):
    return bgr[::-1]


def hex_to_bgr(hex):
    return rgb_to_bgr(hex_to_rgb(hex))


def bgr_to_hex(bgr):
    return rgb_to_hex(bgr_to_rgb(bgr))


if __name__ == '__main__':
    main()
