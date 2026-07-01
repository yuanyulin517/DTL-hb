import time
from contextlib import contextmanager

import torch


class CudaSegTimer:
    """基于 CUDA Event 的分段计时器（低扰动）。

    - 用 `with timer.seg('decode'): ...` 包裹 GPU 区段，事件先入队、不立即同步，
      按 flush_every 周期性 synchronize 后累加，避免逐段 stall GPU。
    - 初始解构造是 CPU 为主，用 construct_start/end（perf_counter）单独计时。
    - enabled=False 时所有方法 no-op，正常训练/评估路径零行为改变。
    """

    def __init__(self, categories, flush_every=200, enabled=True):
        self._cuda = torch.cuda.is_available()
        self.enabled = enabled
        self.acc_ms = {c: 0.0 for c in categories}   # 各 GPU 区段累计耗时(ms)
        self.t_construct_ms = 0.0                    # 初始解构造(ms)
        self.t_total_ms = 0.0                        # 整个 validate 墙钟(ms)
        self._pending = []                           # [(start_event, end_event, category)]
        self.flush_every = flush_every
        self._since_flush = 0
        self._active = True                          # 控制是否记录（用于丢弃第一步）
        self._t0 = None
        self._tc0 = None

    def _sync(self):
        if self._cuda:
            torch.cuda.synchronize()

    def set_active(self, flag):
        self._active = flag

    def begin(self):
        if not self.enabled:
            return
        self._sync()
        self._t0 = time.perf_counter()

    def end(self):
        if not self.enabled:
            return
        self.flush()
        self._sync()
        self.t_total_ms = (time.perf_counter() - self._t0) * 1000.0

    def construct_start(self):
        if not self.enabled:
            return
        self._sync()
        self._tc0 = time.perf_counter()

    def construct_end(self):
        if not self.enabled:
            return
        self._sync()
        self.t_construct_ms = (time.perf_counter() - self._tc0) * 1000.0

    @contextmanager
    def seg(self, category):
        # 未启用或当前步不记录（如第一步预热）→ 正常执行但不计时
        if not (self.enabled and self._active):
            yield
            return
        if self._cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                self._pending.append((start, end, category))
        else:
            t = time.perf_counter()
            try:
                yield
            finally:
                self.acc_ms[category] += (time.perf_counter() - t) * 1000.0

    def step_done(self):
        if not self.enabled:
            return
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.enabled:
            return
        if self._pending:
            self._sync()
            for start, end, cat in self._pending:
                self.acc_ms[cat] += start.elapsed_time(end)
            self._pending.clear()
        self._since_flush = 0


# 共享的禁用计时器：validate 默认不传 timer 时使用，保证零行为改变
NULL_TIMER = CudaSegTimer(['encode', 'decode', 'search'], enabled=False)
