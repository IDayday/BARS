from __future__ import annotations
import csv, os, time
from typing import Any, Dict, Iterable, Optional

class CSVLogger:
    def __init__(self, path: str, default_fields: Optional[Dict[str, Any]] = None):
        self.path = path; self.default_fields = default_fields or {}; os.makedirs(os.path.dirname(path), exist_ok=True); self._fieldnames = None
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, 'r', newline='', encoding='utf-8') as f:
                try: self._fieldnames = next(csv.reader(f))
                except StopIteration: self._fieldnames = None
    def log(self, row: Dict[str, Any]) -> None:
        full = {'time_sec': time.time(), **self.default_fields, **row}
        if self._fieldnames is None:
            self._fieldnames = list(full.keys())
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=self._fieldnames); w.writeheader(); w.writerow({k: _stringify(full.get(k,'')) for k in self._fieldnames})
        else:
            unseen = [k for k in full if k not in self._fieldnames]
            if unseen: self._rewrite_with_extended_header(unseen)
            with open(self.path, 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=self._fieldnames).writerow({k: _stringify(full.get(k,'')) for k in self._fieldnames})
    def _rewrite_with_extended_header(self, new_fields: Iterable[str]) -> None:
        new_header = list(self._fieldnames or []) + list(new_fields); rows = []
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, 'r', newline='', encoding='utf-8') as f: rows = list(csv.DictReader(f))
        with open(self.path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=new_header); w.writeheader()
            for r in rows: w.writerow({k: r.get(k,'') for k in new_header})
        self._fieldnames = new_header

def _stringify(x: Any) -> Any:
    if isinstance(x, (list, tuple, dict)):
        import json; return json.dumps(x, ensure_ascii=False)
    return x

def read_last_csv_row(path: str) -> Dict[str, Any]:
    if not os.path.exists(path) or os.path.getsize(path) == 0: return {}
    with open(path, 'r', newline='', encoding='utf-8') as f:
        last = {}
        for last in csv.DictReader(f): pass
    return last
