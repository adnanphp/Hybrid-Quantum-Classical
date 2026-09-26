"""Track time / CPU / memory / GPU."""
import time
import psutil
import torch
import GPUtil


class ResourceTracker:
    def __init__(self):
        self.start_time = None
        self.start_cpu = None
        self.gpus = GPUtil.getGPUs() if torch.cuda.is_available() else []

    def start(self):
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=None)

    def end(self):
        elapsed = time.time() - self.start_time
        cpu_usage = psutil.cpu_percent(interval=None) - self.start_cpu
        memory_usage = psutil.virtual_memory().used / (1024 ** 3)
        gpu_info = {}
        if self.gpus:
            for gpu in self.gpus:
                gpu_info[f'GPU_{gpu.id}_load'] = gpu.load
                gpu_info[f'GPU_{gpu.id}_mem'] = gpu.memoryUsed
        return {'time_sec': elapsed, 'cpu_usage': cpu_usage,
                'memory_gb': memory_usage, **gpu_info}
