import cv2
import numpy as np

SHOW = True
WRITE = True


def main():
    image = cv2.imread('ame_MiniPixel.png', cv2.IMREAD_UNCHANGED)

    # scale up 10x
    upscaled = cv2.resize(image, (0, 0), fx=10, fy=10,
                          interpolation=cv2.INTER_NEAREST)
    h, w = upscaled.shape[:2]

    # create a background
    reference = cv2.imread('ame_MiniPixel_Display.png', cv2.IMREAD_UNCHANGED)
    hh, ww = reference.shape[:2]

    color = (205, 220, 228)  # BGR
    background = np.full((hh, ww, 3), color, np.uint8)

    offsetX = (hh - w) // 2
    offsetY = (ww - h) // 2

    result = background.copy()
    add_transparent_image(result, upscaled, offsetX, offsetY)

    if WRITE:
        cv2.imwrite('upscaled.png', upscaled)
        cv2.imwrite('result.png', result)

    if SHOW:
        cv2.imshow('original', image)
        cv2.imshow('upscaled', upscaled)
        # cv2.imshow('background', background)
        cv2.imshow('result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


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
