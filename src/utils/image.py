import cv2
import numpy as np
import matplotlib as mpl
from datetime import datetime as dt
from src.utils.centroidtracker import CentroidTracker
from src.utils.save import save_file, save_video

# import cam_v2 as camv
import New_camera_detection as camv

ct = CentroidTracker()

encounter = 0


def letterbox_image(image, size):
    """
    Resize image with unchanged aspect ratio using padding
    """

    # original image size
    ih, iw, ic = image.shape

    # given size
    h, w = size

    # scale and new size of the image
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # placeholder letter box
    new_image = np.zeros((h, w, ic), dtype='uint8') + 128

    # top-left corner
    top, left = (h - nh) // 2, (w - nw) // 2

    # paste the scaled image in the placeholder anchoring at the top-left corner
    new_image[top:top + nh, left:left + nw, :] = cv2.resize(image, (nw, nh))

    return new_image


async def draw_detection(
        img,
        boxes,
        class_names,
        cam_number,
        cls_types,
        kcw,
        consecutive_frames,
        ws,
        codec="MP4V",
        fps=6,
        # drawing configs #

        font=cv2.FONT_HERSHEY_DUPLEX,
        font_scale=0.4,
        box_thickness=1,
        text_color=(255, 255, 255),
        text_weight=1
):
    """
    Draw the bounding boxes on the image
    """

    # encounter = [True] * max_boxes
    # print(encounter)

    global encounter
    act_img = img.copy()
    rectangles = []

    # generate some colors for different classes
    num_classes = len(class_names)  # number of classes
    colors = [mpl.colors.hsv_to_rgb((i / num_classes, 1, 1)) * 255 for i in range(num_classes)]
    # draw the detections
    for count, box in enumerate(boxes):
        rectangles.append(box[:4].astype(int))

        x1, y1, x2, y2 = box[:4].astype(int)
        score = box[-2]
        label = int(box[-1])

        clr = colors[label]

        # draw the bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)

        # text: <object class> (<confidence score in percent>%)
        text = f'{class_names[label]}: {score * 100:.0f}%'

        # get width (tw) and height (th) of the text
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

        # draw the background rectangle
        # img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)
        img = cv2.rectangle(img, (x1, y1 + th), (x1 + tw, y1), clr, -1)
        # put the text
        img = cv2.putText(img, text, (x1, y1 + th), font, font_scale, text_color, text_weight, cv2.LINE_AA)

    objects = ct.update(rectangles)

    for (objectID, centroid,) in objects.items():
        if encounter == objectID:
            encounter += 1
            for count, box in enumerate(boxes):

                # для сохранения только первого появление
                label = int(box[-1])

                if class_names[label] in cls_types:

                    x1, y1, x2, y2 = box[:4].astype(int)

                    save_time = dt.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
                    consecutive_frames = 0

                    # сохранить фрейм со всеми боксами
                    full_img = img

                    await camv.send_image(cv2.imencode('.jpg', act_img[y1:y2, x1:x2])[1],
                                          cv2.imencode('.jpg', full_img)[1],
                                          cv2.imencode('.jpg', act_img)[1],
                                          save_time,
                                          class_names[label],
                                          objectID,
                                          ws)
                    save_file(act_img[y1:y2, x1:x2], full_img, act_img, save_time, class_names[label], objectID,
                              cam_number)

                    # for i in kcw:
                    if not kcw[count].recording:
                        save_video(save_time, class_names[label], objectID, cam_number, kcw[count], codec, fps)

                    # # log
                    # if class_names[label] == 'Person':
                    #     print(f'event {objectID}. person without hat and vest. {save_time}')
                    # elif class_names[label] == 'Hat':
                    #     print(f'event {objectID}. person only with hat. {save_time}')
                    # elif class_names[label] == 'Vest':
                    #     print(f'event {objectID}. person only with vest. {save_time}')
                    # else:
                    #     print(f'event {objectID}. person with hat and vest. {save_time}')

    return img, consecutive_frames
