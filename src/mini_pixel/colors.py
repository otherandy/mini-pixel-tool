def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


def rgb_to_bgr(rgb):
    return rgb[::-1]


def bgr_to_rgb(bgr):
    return bgr[::-1]


def hex_to_bgr(hex):
    return rgb_to_bgr(hex_to_rgb(hex))


def bgr_to_hex(bgr):
    return rgb_to_hex(bgr_to_rgb(bgr))
