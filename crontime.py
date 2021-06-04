import pycron
import time
from src.utils.delete_until import check_memory

while True:
    if pycron.is_now('0 0 * * *'):
        # print('checking memory')
        check_memory()
        time.sleep(180)
    elif pycron.is_now('0 12 * * *'):
        # print('checking memory')
        check_memory()
        time.sleep(180)
    else:
        # print('waiting')
        time.sleep(30)
