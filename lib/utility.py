import cv2
import torch


def load_image(image_path: str, image_size: int or list) -> torch.Tensor:
    """
    Process the image for easier use
    :param image_path: the path to the image
    :param image_size: the size of the image wanted
    :return:
    """
    img1 = cv2.imread(image_path)
    if isinstance(image_size, int):
        img1 = cv2.resize(img1, tuple([image_size, image_size]))
    else:
        img1 = cv2.resize(img1, tuple(image_size))
    img1 = torch.tensor(img1 / 255, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return img1
