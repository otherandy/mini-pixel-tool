import cv2
import numpy as np
import os
import argparse
import textwrap
from pathlib import Path
from mini_pixel.colors import bgr_to_hex, hex_to_bgr

# default values
SCALE = 10
BACKGROUND_COLOR = bgr_to_hex((205, 220, 228))
DISPLAY_SIZE = 500
OUTPUT_DIR = '.'
UPSCALE_DIR = 'Scaled'
DISPLAY_DIR = 'Display'

DEBUG = False


def main():
    parser = argparse.ArgumentParser(
        prog='mini_pixel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            Mini Pixel Utility Tool
            -----------------------

            Upscales a mini pixel, creates a new display image and adds a shadow
            (requires a_Shadow2_MiniPixelDisplay.png).
            '''),
        epilog=f'Example: mini_pixel input.png -s {SCALE} -c {BACKGROUND_COLOR} -d {DISPLAY_SIZE} -o out --no-shadow --no-write --show')

    parser.add_argument('input_filepath', type=str,
                        help='path to the input image')
    parser.add_argument('-s', type=int, default=SCALE,
                        dest='scale',
                        required=False,
                        help=f'scale factor (default: {SCALE})')
    parser.add_argument('-c', type=str, default=BACKGROUND_COLOR,
                        dest='background_color',
                        required=False,
                        help=f'background color in hex (default: {BACKGROUND_COLOR})')
    parser.add_argument('-d', type=int, default=DISPLAY_SIZE,
                        dest='display_size',
                        required=False,
                        help=f'final size of display image in pixels (default: {DISPLAY_SIZE})')
    parser.add_argument('-o', type=str, default=OUTPUT_DIR,
                        dest='output_dir',
                        required=False,
                        help=f'output directory (default: {OUTPUT_DIR})')

    parser.add_argument('--no-shadow', action='store_true',
                        help='do not add a shadow')

    parser.add_argument('--flat-write', action='store_true',
                        help='write the result to the output directory without creating subdirectories')
    parser.add_argument('--sdir', type=str, default=UPSCALE_DIR,
                        dest='scaled_dir',
                        required=False,
                        help=f'scaled output subdirectory (default: {UPSCALE_DIR})')
    parser.add_argument('--ddir', type=str, default=DISPLAY_DIR,
                        dest='display_dir',
                        required=False,
                        help=f'display output subdirectory (default: {DISPLAY_DIR})')

    parser.add_argument('--no-write', action='store_true',
                        help='do not write the result to the output directory')
    parser.add_argument('--show', action='store_true',
                        help='show the result, press any key to close the windows')
    args = parser.parse_args()

    input_filepath = args.input_filepath
    scale = args.scale
    background_color = hex_to_bgr(args.background_color)
    display_size = args.display_size
    add_shadow = not args.no_shadow

    output_dir = args.output_dir

    flat_write = args.flat_write
    upscale_dir = args.scaled_dir if not flat_write else ''
    display_dir = args.display_dir if not flat_write else ''

    write = not args.no_write
    show = args.show

    run(input_filepath, scale, background_color, display_size, add_shadow,
        output_dir, upscale_dir, display_dir, write, flat_write, show)


def run(input_filepath, scale=SCALE,
        background_color=BACKGROUND_COLOR, display_size=DISPLAY_SIZE,
        output_dir=OUTPUT_DIR, upscale_dir=UPSCALE_DIR, display_dir=DISPLAY_DIR,
        add_shadow=True, write=True, show=False):
    if input_filepath is None:
        input_filepath = input('Please provide an input file path: ')

    original = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)
    if original is None:
        print(f'Error: could not read input image {input_filepath}')
        return

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
        add_shadow_image(result, offsetX, offsetY,
                         scale, original.shape[0])
    add_transparent_image(result, upscaled, offsetX, offsetY)

    # result output
    if write:
        write_images(upscaled, result, input_filepath, scale,
                     output_dir, upscale_dir, display_dir)
    if show:
        show_images(original, upscaled, result)


def write_images(upscaled, result, input_filepath, scale,
                 output_dir, upscale_dir, display_dir):
    # create diretories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(os.path.join(output_dir, upscale_dir)):
        os.makedirs(os.path.join(output_dir, upscale_dir))

    if not os.path.exists(os.path.join(output_dir, display_dir)):
        os.makedirs(os.path.join(output_dir, display_dir))

    # write upscaled image
    p = Path(input_filepath)
    upscaled_path = os.path.join(
        output_dir, upscale_dir, f'{p.stem}x{scale}{p.suffix}')
    if not cv2.imwrite(upscaled_path, upscaled):
        print(f'Error: could not write upscaled image to {upscaled_path}')

    # write display image
    display_path = os.path.join(
        output_dir, display_dir, f'{p.stem}_Display{p.suffix}')
    if not cv2.imwrite(display_path, result):
        print(f'Error: could not write display image to {display_path}')


def show_images(image, upscaled, result):
    cv2.imshow('original', image)
    cv2.imshow('upscaled', upscaled)
    cv2.imshow('result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_shadow_image(background, offsetX, offsetY, scale, y):
    shadow = cv2.imread(os.path.join(os.path.dirname(__file__),
                        'a_Shadow2_MiniPixelDisplay.png'), cv2.IMREAD_UNCHANGED)
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


# https://stackoverflow.com/a/71701023
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


if __name__ == '__main__':
    main()
