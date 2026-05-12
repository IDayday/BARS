from __future__ import annotations
import signal, threading
from typing import Optional
class Stopper:
    def __init__(self): self._event = threading.Event(); self.reason: Optional[str] = None
    def request_stop(self, reason: str = 'signal') -> None: self.reason = reason; self._event.set()
    @property
    def stop_requested(self) -> bool: return self._event.is_set()
GLOBAL_STOPPER = Stopper()
def install_signal_handlers(stopper: Stopper = GLOBAL_STOPPER) -> None:
    def _handler(signum, frame): stopper.request_stop(f'signal_{signum}')
    signal.signal(signal.SIGTERM, _handler); signal.signal(signal.SIGINT, _handler)
