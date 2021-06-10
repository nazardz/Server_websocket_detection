#!/usr/bin/env python
import json
from imutils.video import VideoStream
import imutils
from src.utils.fixes import *
from src.utils.image import *
from src.yolo3.detect import *
from src.yolo3.model import *
import base64
from src.utils.video_writer import KeyClipWriter
import websockets
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

fix_tf_gpu()

CLASS_TYPES = ['Person', 'Hat', 'Vest', 'Hat_Vest']
THRESHOLD = 0.8
MAX_BOXES = 10

codec = "mp4v"
buffer_size = 27
fps = 5

# On/off detection
DETECTION_CAM = 1
# run detection
RUN_PROC = 1


def proc_start(cam_rtsp, serv_address, cam_number):
    global SERV_ADDRESS, RTSP, CAM_NUMBER
    RTSP = cam_rtsp
    SERV_ADDRESS = serv_address
    CAM_NUMBER = cam_number
    asyncio.run(start_detection_on_cam())


def prepare_model():
    """ Prepare the YOLO model """
    global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model

    # shape (height, width) of the input image
    input_shape = (416, 416)

    # class names
    class_names = ['Person', 'Hat', 'Vest', 'Hat_Vest']

    # anchor boxes
    anchor_boxes = np.array(
        [
            np.array([[73, 158], [128, 209], [224, 246]]) / 32,  # output-1 anchor boxes
            np.array([[32, 50], [40, 104], [76, 73]]) / 16,  # output-2 anchor boxes
            np.array([[6, 11], [11, 23], [19, 36]]) / 8  # output-3 anchor boxes
        ],
        dtype='float64'
    )

    # number of classes and number of anchors
    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]

    # input and output
    input_tensor = Input(shape=(input_shape[0], input_shape[1], 3))  # input
    num_out_filters = (num_anchors // 3) * (5 + num_classes)  # output

    # build the model
    model = yolo_body(input_tensor, num_out_filters)

    # load weights
    weight_path = './model-data/weights/pictor-ppe-v302-a2-yolo-v3-weights.h5'
    model.load_weights(weight_path)


async def get_detection(img, kcw, consecutive_frames, ws):
    # save a copy of the img
    act_img = img.copy()

    # shape of the image
    ih, iw = act_img.shape[:2]

    # save max boxes for this cycle
    max_boxes_last = MAX_BOXES

    # preprocess the image
    img = letterbox_image(img, input_shape)
    img = np.expand_dims(img, 0)
    image_data = np.array(img) / 255.

    # raw prediction from yolo model
    prediction = model.predict(image_data)

    # process the raw prediction to get the bounding boxes
    boxes = detection(
        prediction,
        anchor_boxes,
        num_classes,
        image_shape=(ih, iw),
        input_shape=(416, 416),
        max_boxes=max_boxes_last,
        score_threshold=THRESHOLD,
        iou_threshold=0.45,
        classes_can_overlap=False)

    # convert tensor to numpy
    boxes = boxes[0].numpy()

    # draw the detection on the actual image
    return await draw_detection(img=act_img,
                                boxes=boxes,
                                class_names=class_names,
                                cam_number=CAM_NUMBER,
                                cls_types=CLASS_TYPES,
                                kcw=kcw,
                                consecutive_frames=consecutive_frames,
                                codec=codec,
                                fps=fps,
                                ws=ws
                                )


async def start_detection_on_cam():
    global DETECTION_CAM
    prepare_model()

    async with websockets.connect(SERV_ADDRESS) as websocket:
        # data_join = json.dumps({"event": "join_camera_client"})
        # await websocket.send(data_join)
        msg = json.dumps({"event": "detection_start", "source": RTSP})
        await websocket.send(msg)

        logging.info(f'{RTSP} start stream...')
        vs = VideoStream(RTSP).start()
        kcw = []
        for i in range(MAX_BOXES):
            kcw.append(KeyClipWriter(bufSize=buffer_size))
        consecutive_frames = 0
        while RUN_PROC:

            try:
                await message_handler(websocket)
            except asyncio.TimeoutError:
                pass

            frame = vs.read()
            if DETECTION_CAM:
                frame, consecutive_frames = await get_detection(imutils.resize(frame, width=800), kcw,
                                                                consecutive_frames,
                                                                websocket)
            consecutive_frames += 1
            for i in range(len(kcw)):
                kcw[i].update(frame)
                if kcw[i].recording and consecutive_frames == buffer_size:
                    kcw[i].finish()

        # ####
        # show frames
        #     cv2.imshow(f'Cam #{CAM_NUMBER}', imutils.resize(frame, width=800))
        #     pressed_key = cv2.waitKey(1) & 0xFF
        #     # on/off recognition
        #     if pressed_key == ord('d'):
        #         if DETECTION_CAM:
        #             print(f'DETECTION OFF on Cam - {CAM_NUMBER}')
        #             DETECTION_CAM = False
        #         else:
        #             print(f'DETECTION ON  on Cam - {CAM_NUMBER}')
        #             DETECTION_CAM = True
        #     # close detection
        #     elif pressed_key == ord('q'):
        #         print(f'{CAM_NUMBER} quit stream.')
        #         break
        #
        # cv2.destroyWindow(RTSP)
        # ####
        logging.info(f'{RTSP} end stream...')
        msg = json.dumps({"event": "detection_end", "source": RTSP})
        await websocket.send(msg)
        vs.stop()


async def send_image(img_box, img_full, img_clean, save_datetime, class_name, object_id, ws):
    encode_box = base64.b64encode(img_box).decode()
    encode_full = base64.b64encode(img_full).decode()
    encode_clean = base64.b64encode(img_clean).decode()
    screenshot_box = f'data:image/jpg;base64,{encode_box}'
    screenshot_full = f'data:image/jpg;base64,{encode_full}'
    screenshot_clean = f'data:image/jpg;base64,{encode_clean}'
    # представление данных
    data = {
        "event_id": object_id,
        "time": save_datetime,
        "type": class_name,
        "screenshot": [screenshot_box, screenshot_full, screenshot_clean],
        "source": RTSP
    }
    # отправка данных
    data = json.dumps({"event": "object_detected", "data": data})
    await message_producer(ws, data)


async def message_handler(ws):
    global MAX_BOXES, CLASS_TYPES, DETECTION_CAM, THRESHOLD, RUN_PROC
    message = await asyncio.wait_for(ws.recv(), timeout=0.1)
    message = json.loads(message)
    event = message['event']
    response = json.dumps({'event': 'response', 'rec': event, 'status': True, 'source': RTSP})
    if event == 'switch_detection':
        logging.info(f'changed detection from {DETECTION_CAM} to {message["data"]}')
        DETECTION_CAM = message['data']
        await message_producer(ws, response)
    elif event == 'change_max_boxes':
        logging.info(f'changed max boxes from {MAX_BOXES} to {message["data"]}')
        MAX_BOXES = message['data']
        await message_producer(ws, response)
    elif event == 'change_treshold':
        logging.info(f'changed threshold from {THRESHOLD} to {message["data"]}')
        THRESHOLD = message['data']
        await message_producer(ws, response)
    elif event == 'change_detection_type':
        logging.info(f'changed class types from {CLASS_TYPES} to {message["data"]}')
        CLASS_TYPES = message['data']
        await message_producer(ws, response)
    elif event == 'finish_detection':
        RUN_PROC = False
    else:
        response = json.dumps({'event': 'response', 'rec': event, 'status': False, 'source': RTSP})
        await message_producer(ws, response)


async def message_producer(ws, data):
    await ws.send(data)
    # logging.info(f'{SERV_ADDRESS} send message to')
