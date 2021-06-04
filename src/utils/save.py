import os
import cv2
from src.utils.delete_until import get_mount_points, check_memory


def save_file(img_box, img_full, img_clean, save_datetime, class_name, object_id, camera_number):
    """сохранение скриншотов"""
    try:
        IMG_path = get_mount_points()[0][1]
        # split_date = save_datetime.rsplit('_', maxsplit=4)
        img_save_path = f'{IMG_path}/cam_{camera_number}/event_{object_id}/{save_datetime}/'
        os.makedirs(img_save_path, exist_ok=True)
        cv2.imwrite(f'{img_save_path}/{save_datetime}_{class_name}_{object_id}_box.jpg', img_box)
        cv2.imwrite(f'{img_save_path}/{save_datetime}_{object_id}_full.jpg', img_full)
        cv2.imwrite(f'{img_save_path}/{save_datetime}_{object_id}_clean.jpg', img_clean)
    except Exception as e:
        print('[ERROR]: ', e)


def save_video(save_datetime, class_name, object_id, camera_number, kcw, codec, fps):
    """сохранение видеофайла"""
    try:
        # получить путь к флешке
        IMG_path = get_mount_points()[0][1]
        # split_date = save_datetime.rsplit('_', maxsplit=4)
        # проверка на класс (hat/vest) сохранения
        img_save_path = f'{IMG_path}/cam_{camera_number}/event_{object_id}/{save_datetime}/'
        os.makedirs(img_save_path, exist_ok=True)
        kcw.start(f"{img_save_path}/{save_datetime}_{class_name}_{object_id}.mp4",
                  cv2.VideoWriter_fourcc(*codec), fps)
    except Exception as e:
        print('[ERROR]: ', e)
