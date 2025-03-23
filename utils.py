import os
import io
from pathlib import Path
from PIL import Image
import settings


def make_save_dir():
    """
    make save directory for output image
    """
    os.makedirs(Path(settings.BASE_DIR, settings.SAVE_DIR_NAME), exist_ok=True)
    return Path(settings.BASE_DIR, settings.SAVE_DIR_NAME)


def get_image_files_list(folder: str) -> list:
    all_files = os.listdir(folder)
    image_files = []
    for file in all_files:
        extension = file.lower().split(".")[-1]
        if extension in ["jpg", "png", "jpeg", "jpe"]:
            image_files.append(file)
    image_files.sort()
    return image_files


def get_image_by_file_name(filename: str) -> Image:
    return Image.open(filename)


def get_image_by_kuwahara_filter(filtered_pic) -> Image:
    return Image.fromarray(filtered_pic)


def get_resized_image(im, height: int, width: int) -> Image:
    height = round(height * (settings.MAX_PICTURE_WIDTH / width))
    return im.resize((settings.MAX_PICTURE_WIDTH, height))


def get_picture_bytes_data(im: Image) -> bytes:
    height, width = im.height, im.width
    if width > settings.MAX_PICTURE_WIDTH:
        im = get_resized_image(im, height, width)
    im_bytes = io.BytesIO()
    im.save(im_bytes, format="PNG")
    return im_bytes.getvalue()
