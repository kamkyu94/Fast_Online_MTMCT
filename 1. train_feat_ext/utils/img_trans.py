import cv2
import config


# Resize a rectangular image to a padded rectangular
def letterbox(img, color=(0, 0, 0)):
    # shape = [height, width]
    shape = img.shape[:2]
    ratio = min(float(config.img_h) / shape[0], float(config.img_w) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))

    # Padding
    dh = (config.img_h - new_shape[1]) / 2
    dw = (config.img_w - new_shape[0]) / 2

    # Top, bottom, left, right
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border, padded rectangular
    image = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image
