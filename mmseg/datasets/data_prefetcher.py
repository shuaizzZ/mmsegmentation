
from typing import *
from queue import Queue, Full, Empty
from time import sleep
import torch
from torch.utils.data import DataLoader
from threading import Thread


__all__ = ["DataPrefetcher", "CudaDevice"]


class CudaDevice(object):
    def __init__(self, gpu_str: Text):
        self.gpu_str = gpu_str

    @property
    def gpu_str(self):
        return self._gpu_str

    @gpu_str.setter
    def gpu_str(self, gpu_str: Text):
        if not isinstance(gpu_str, str):
            raise TypeError("Expect str but got {}.".format(gpu_str))
        self._gpu_str = gpu_str
        self.device_ids = list(map(int, self.gpu_str.split(",")))
        self.device = torch.device(device="cuda:{}".format(",".join(map(str, self.device_ids))))

    def __str__(self):
        return self.device.__str__()

    def __repr__(self):
        return self.device.__repr__()


class _DataPrefetcherCPU(Thread):
    def __init__(self,
                 dataloader: DataLoader,
                 queue: Queue,
                 frequency: float = 0.05,
                 timeout: float = 9):
        super().__init__()
        self.dataloader = dataloader
        self.queue = queue
        self.frequency = frequency
        self.timeout = timeout
        self.__to_close = False
        self.start()

    @property
    def dataloader(self):
        return self._dataloader

    @dataloader.setter
    def dataloader(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.iterator = iter(self.dataloader)

    def run(self):
        while True:
            if self.__to_close:
                break
            if self.queue.full():
                sleep(self.frequency)
            else:
                try:
                    item = next(self.iterator)
                    self.queue.put(item, timeout=self.timeout)
                except StopIteration:
                    break
                except Exception as e:
                    raise e

    def close(self):
        self.__to_close = True
        self.queue.queue.clear()
        self.join()


class DataPrefetcher(Thread):
    """
    Examples:
        for epoch in range(epochs):
            data_prefetcher = DataPrefetcher(dataloader)
            for batch_count, values in data_prefetcher:
                pass
    """
    def __init__(self,
                 dataloader: DataLoader,
                 queue_max_size: int = 2,
                 cuda_device: CudaDevice = None,
                 frequency: float = 0.1,
                 timeout: float = 9,
                 *args, **kwargs):
        """DataPrefetcher needs to be initialized every epoch.

        Args:
            dataloader (DataLoader):
            queue_max_size (int, optional):
            cuda_device (CudaDevice, optional):
            frequency (float, optional):
            timeout (float, optional):
            *args:
            **kwargs:
        """
        super().__init__()
        self.dataloader = dataloader
        self.count = len(self)
        self.queue_max_size = queue_max_size
        self.cuda_device = cuda_device
        self.frequency = frequency
        self.timeout = timeout
        self.keys = ['img', 'gt_semantic_seg']

        self.queue = Queue(self.queue_max_size)
        self.cpu_queue = Queue(self.queue_max_size)
        self.cpu_fetcher = _DataPrefetcherCPU(self.dataloader, self.cpu_queue, self.frequency, self.timeout)
        self.__to_close = False
        self.stream = torch.cuda.Stream()
        self.start()

    def run(self):
        with torch.cuda.stream(self.stream):
            while True:
                if self.__to_close:
                    self.queue.queue.clear()
                    break
                if self.queue.full():
                    sleep(self.frequency)
                else:
                    try:
                        item = self.cpu_queue.get(timeout=self.timeout)
                        # for key in self.keys:
                        #     for i, data in enumerate(item[key]):
                        #         item[key][i] = data.cuda(non_blocking=True)
                        self.queue.put(item)
                    except Empty:
                        break
                    except Exception as e:
                        raise e

    def __iter__(self):
        return self

    def __len__(self):
        return self.dataloader.__len__()

    def __next__(self):
        if not self.count:
            self.close()
            raise StopIteration
        try:
            item = self.queue.get(timeout=self.timeout)
            self.count -= 1
            return item
        except Empty as e:
            Warning(e.message)
            self.close()
            raise StopIteration

    def close(self):
        self.cpu_fetcher.close()
        self.__to_close = True
        self.join()

    @property
    def batch_size(self):
        return self.dataloader.batch_size

    @property
    def dataset(self):
        return self.dataloader.dataset