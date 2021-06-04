import os
from glob import glob
from subprocess import check_output
import psutil
import shutil


def get_usb_devices():
    # if devices are None: get_usb_devices
    sdb_devices = map(os.path.realpath, glob('/sys/block/sd*'))
    usb_devices = (dev for dev in sdb_devices
                   if any(['usb' in dev.split('/')[5],
                           'usb' in dev.split('/')[6]]))
    return dict((os.path.basename(dev), dev) for dev in usb_devices)


def get_mount_points(devices=None):
    """ получить путь до флешки """
    devices = devices or get_usb_devices()  # if devices are None: get_usb_devices
    output = check_output(['mount']).splitlines()
    output = [tmp.decode('UTF-8') for tmp in output]

    def is_usb(path):
        return any(dev in path for dev in devices)

    usb_info = (line for line in output if is_usb(line.split()[0]))

    full_info = []
    for info in usb_info:
        mount_URI = info.split()[0]
        usb_URI = info.split()[2]
        for x in range(3, info.split().__sizeof__()):
            if info.split()[x].__eq__("type"):
                for m in range(3, x):
                    usb_URI += " " + info.split()[m]
                break
        full_info.append([mount_URI, usb_URI])
    return full_info


def get_files_with_size(dir_name):
    file_paths = filter(os.path.isdir, [os.path.join(dir_name, basename) for basename in os.listdir(dir_name)])
    # сохранить в список путь файлов/папок
    file_paths = [(filepath, os.path.getsize(filepath)) for filepath in file_paths]
    # file_paths = [filepath for filepath in file_paths]
    # сортровать по времени
    file_paths.sort(key=lambda filepath: os.path.getmtime(filepath[0]))
    return file_paths


def check_memory(path=None, capacity=None):
    """проверка заполненности памяти: путь до флешки и процент заполненности"""
    if path is None:
        path = get_mount_points()[0][1]
    if capacity is None:
        capacity = 90.0
    if len(path) != 0:
        # пероверка вставлена ли флешка
        print(capacity)
        print(path)
        obj_disk = psutil.disk_usage(path)
        print(f"Использовано: {obj_disk.percent}%")
        print(f"Свободно: {100 - obj_disk.percent}%")
        files = get_files_with_size(path)
        # вывод информаций про флешку,

        # если свободного места меньше указанного то удалит файлы/папки на флешке, начиная с самой старой
        i = 0
        try:
            while obj_disk.percent >= capacity:  # CAPACITY = 90.0
                file_p = files[i]
                # пропуск файла
                if file_p[0].__contains__('.Trash-1000'):
                    i += 1
                    file_p = files[i]
                print(f"Удаляю : {file_p[0]}")
                shutil.rmtree(file_p[0], ignore_errors=True)
                # удаление папки/файла
                obj_disk = psutil.disk_usage(path)
                # обновление информаций про флешку
                i += 1

        except Exception as e:
            print(e)
        finally:
            print(f"Использовано: {obj_disk.percent}%")
            print(f"Свободно: {100 - obj_disk.percent}%")
    else:
        print('вставьте флешку')

# TODO stop saving
# создать условия для остановки записи на флешку
# TODO full format
# создать условия для полного форматирования флешки
