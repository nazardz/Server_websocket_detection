#!/usr/bin/env python

import asyncio
import websockets
from websockets import WebSocketServerProtocol
import json
import multiprocessing as mp
from New_camera_detection import proc_start
from src.utils.delete_until import check_memory
import logging
logging.basicConfig(level=logging.INFO)

HOST = '0.0.0.0'
PORT = 8080
pr_list = []
cam_counter = 0

# TODO
# группировка камер,
# вернуть адресс камер+название на сервер и передать их клиенту
# проверка своюодного места
# отправка видео файла


class Server:
    sender_client = None
    camera_clients = set()
    all_clients = set()
    response = []

    # ругистр всех клиентов подключенных к серверу
    async def register(self, ws: WebSocketServerProtocol):
        self.all_clients.add(ws)
        logging.info(f'{ws.remote_address} connects.')

    # отвязка отключенных клиентов
    async def unregister(self, ws: WebSocketServerProtocol):
        # если клиент - это клиент поднявший камеры, то отключить все камеры
        if ws == self.sender_client:
            self.all_clients.remove(ws)
            logging.info(f'{ws.remote_address} disconnects.')
            data_join = json.dumps({"event": "finish_detection"})
            for client in self.all_clients:
                await client.send(data_join)
                logging.info(f'{client.remote_address} disconnects.')
            self.all_clients.clear()
        else:
            if len(self.all_clients) != 0:
                self.all_clients.remove(ws)
                logging.info(f'{ws.remote_address} disconnects.')

    # отправка команды на камеру
    async def send_to_camera(self, data=None):
        data = json.dumps(data)
        for client in self.camera_clients:
            await client.send(data)
            logging.info(f'{client.remote_address} send message to.')
        # if self.camera_clients:
        #     for client in self.camera_clients:
        #         await client.send(data)

    # отправка сообщения на клиент поднявшего камеру
    async def send_to_sender(self, data):
        data = json.dumps(data)
        await self.sender_client.send(data)
        logging.info(f'{self.sender_client.remote_address} send message to.')

    # обработчик сообщении
    async def distribute(self, ws: WebSocketServerProtocol):
        async for message in ws:
            # перенос команды в json
            data = json.loads(message)
            # сохранения события
            event = data['event']
            logging.info(f'{ws.remote_address} receive message from.')
            res = json.dumps({"status": True})

            # сообщение камере -->
            if event == 'start_detection':
                self.sender_client = ws
                await start_detection(data)
                await ws.send(res)
            elif event == 'switch_detection':
                if self.sender_client is not None:
                    await self.send_to_camera(data)
                else:
                    res = json.dumps({"status": False})
                    await ws.send(res)
            elif event == 'change_treshold':
                if self.sender_client is not None:
                    await self.send_to_camera(data)
                else:
                    res = json.dumps({"status": False})
                    await ws.send(res)
            elif event == 'change_max_boxes':
                if self.sender_client is not None:
                    await self.send_to_camera(data)
                else:
                    res = json.dumps({"status": False})
                    await ws.send(res)
            elif event == 'change_detection_type':
                if self.sender_client is not None:
                    await self.send_to_camera(data)
                else:
                    res = json.dumps({"status": False})
                    await ws.send(res)
            elif event == 'finish_detection':
                if self.sender_client is not None:
                    await self.send_to_camera(data)
                else:
                    res = json.dumps({"status": False})
                    await ws.send(res)
            # ответ клиенту
            elif event == 'object_detected':
                await self.send_to_sender(data)
            elif event == 'detection_start':
                self.camera_clients.add(ws)
                logging.info(f'{ws.remote_address} add to camera list.')
                # if cam_counter == len(self.camera_clients):
                # await self.send_to_sender({'result': self.camera_clients})
            elif event == 'detection_end':
                self.camera_clients.remove(ws)
                logging.info(f'{ws.remote_address} remove from camera list.')
            elif data['event'] == 'response':
                del data['event']
                data.update({'event': data['rec']})
                del data['rec']
                self.response.append(data)
                if len(self.response) >= len(self.camera_clients):
                    # send response
                    await self.send_to_sender({'result': self.response})
                    self.response = []
                    logging.info(f'send response from cameras.')

            # если event не соответствует не одному из команд
            else:
                res = json.dumps({"error": f"Invalid message. Event <{event}> doesn't exist"})
                await ws.send(res)

    # обработчик сервера
    async def ws_handler(self, ws: WebSocketServerProtocol, url):
        await self.register(ws)
        try:
            await asyncio.sleep(1)
            await self.distribute(ws)
        except websockets.exceptions.ConnectionClosedError:
            pass
        finally:
            await self.unregister(ws)


# запуск камер на новых процессах чтобы не блокировать сервер
async def start_detection(data):
    """начать детекцию с полученными rtsp ссылками"""
    global cam_counter
    serv_address = f'ws://{HOST}:{str(PORT)}'
    for cam_rtsp in data['data']:
        cam_counter += 1
        p = mp.Process(target=proc_start,
                       args=(cam_rtsp, serv_address, cam_counter),
                       daemon=True)
        p.start()
        logging.info(f'start detection on cam #{cam_counter}.')


# запуск сервера
if __name__ == "__main__":
    start_server = websockets.serve(Server().ws_handler, HOST, PORT)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
    loop.run_forever()
