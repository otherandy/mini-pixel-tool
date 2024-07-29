import cv2
import numpy as np
import os

SHOW = False
WRITE = True
DEBUG = False

SCALE = 10
BACKGROUND_COLOR = (205, 220, 228)  # BGR

INPUT_FILE = 'ame_MiniPixel.png'
OUTPUT_DIR = './out'


def main():
    image = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)

    # scale up the image
    upscaled = cv2.resize(image, (0, 0), fx=SCALE, fy=SCALE,
                          interpolation=cv2.INTER_NEAREST)
    h, w = upscaled.shape[:2]

    # create a background
    reference = cv2.imread('ame_MiniPixel_Display.png', cv2.IMREAD_UNCHANGED)
    hh, ww = reference.shape[:2]
    offsetX = (hh - w) // 2
    offsetY = (ww - h) // 2

    shadow = cv2.imread('a_Shadow2_MiniPixelDisplay.png', cv2.IMREAD_UNCHANGED)
    _, y = find_lower_right_pixel(image)
    shadowOffsetX = offsetX - SCALE
    shadowOffsetY = offsetY + y * SCALE

    if DEBUG:
        print(f'shadowOffset: ({shadowOffsetX}, {shadowOffsetY})')

    result = np.full((hh, ww, 3), BACKGROUND_COLOR, np.uint8)
    add_transparent_image(result, shadow, shadowOffsetX, shadowOffsetY)
    add_transparent_image(result, upscaled, offsetX, offsetY)

    if WRITE:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'upscaled.png'), upscaled)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'result.png'), result)

    if SHOW:
        cv2.imshow('original', image)
        cv2.imshow('upscaled', upscaled)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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


if __name__ == '__main__':
    main()
