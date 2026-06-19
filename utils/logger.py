"""Timestamped text logger for evaluation runs."""

import os
from datetime import datetime


class EvalLogger:
    """Write evaluation messages to a timestamped log file."""

    def __init__(self, log_dir, prefix="eval"):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"{prefix}_{timestamp}.txt")
        self._file = open(self.path, "w", encoding="utf-8")
        self.log(f"Evaluation log")
        self.log(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Log file: {self.path}")
        self.log("")

    def log(self, msg=""):
        self._file.write(f"{msg}\n")
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
