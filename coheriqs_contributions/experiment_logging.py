"""Process-wide logging policy for contribution experiment runners."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
import sys


class EventCategory(str, Enum):
    """Stable semantic categories used by experiment log records."""

    STATUS = "status"
    CONFIG = "config"
    INITIAL_DETAILS = "initial_details"
    CWT = "cwt"
    EPOCH = "epoch"
    TRAIN_STEP = "train_step"
    SELECTOR = "selector"
    MOABB = "moabb"
    FINAL_RESULTS = "final_results"
    ARTIFACT = "artifact"


@dataclass(frozen=True)
class ConsolePolicy:
    """Console visibility settings independent from durable file logging."""

    initial_details: bool = False
    final_results: bool = True
    cwt_progress: bool = False
    moabb_progress: bool = False
    train_steps: bool = False
    epoch_every: int = 5
    selector_every: int = 0

    def __post_init__(self) -> None:
        if self.epoch_every < 0:
            raise ValueError("console epoch cadence must be non-negative.")
        if self.selector_every < 0:
            raise ValueError("console selector cadence must be non-negative.")


_console_policy: ConsolePolicy | None = None


def get_console_policy() -> ConsolePolicy | None:
    """Return the configured process policy, if an experiment runner set one."""

    return _console_policy


def is_experiment_logging_configured() -> bool:
    """Whether the current process has an active experiment logging policy."""

    return _console_policy is not None


def resolve_experiment_log_path(
    requested_path: str | os.PathLike[str] | None,
    *,
    run_id: str,
    moabb_results_root: str | os.PathLike[str],
) -> Path:
    """Resolve a collision-safe experiment log path for an experiment run."""

    if requested_path is not None:
        log_path = Path(requested_path).expanduser().resolve()
    else:
        log_path = (
            Path(moabb_results_root).expanduser().resolve()
            / run_id
            / "experiment.log"
        )

    if log_path.exists():
        raise FileExistsError(
            f"Experiment log already exists; refusing to overwrite: {log_path}"
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


class _SemanticConsoleFilter(logging.Filter):
    def __init__(self, policy: ConsolePolicy) -> None:
        super().__init__()
        self.policy = policy

    @staticmethod
    def _at_cadence(record: logging.LogRecord, cadence: int) -> bool:
        if cadence <= 0:
            return False
        epoch = getattr(record, "event_epoch", None)
        total = getattr(record, "event_total_epochs", None)
        if not isinstance(epoch, int):
            return False
        return epoch % cadence == 0 or (isinstance(total, int) and epoch == total)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True

        category = getattr(record, "event_category", None)
        if isinstance(category, EventCategory):
            category = category.value
        if category == EventCategory.ARTIFACT.value:
            return True
        if category in {EventCategory.STATUS.value, EventCategory.CONFIG.value}:
            return True
        if category == EventCategory.INITIAL_DETAILS.value:
            return self.policy.initial_details
        if category == EventCategory.FINAL_RESULTS.value:
            return self.policy.final_results
        if category == EventCategory.CWT.value:
            return self.policy.cwt_progress
        if category == EventCategory.MOABB.value:
            return self.policy.moabb_progress
        if category == EventCategory.EPOCH.value:
            return self._at_cadence(record, self.policy.epoch_every)
        if category == EventCategory.TRAIN_STEP.value:
            return self.policy.train_steps
        if category == EventCategory.SELECTOR.value:
            return self._at_cadence(record, self.policy.selector_every)
        return False


class _DurableCategoryFilter(logging.Filter):
    """Give third-party records a stable category before either handler sees them."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "event_category"):
            record.event_category = (
                EventCategory.MOABB.value
                if record.name == "moabb" or record.name.startswith("moabb.")
                else "external"
            )
        return True


def configure_experiment_logging(
    log_path: Path,
    *,
    console_policy: ConsolePolicy,
) -> None:
    """Install one durable file handler and one semantic console handler."""

    global _console_policy

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(_DurableCategoryFilter())
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s "
            "[%(event_category)s] %(message)s",
            defaults={"event_category": "external"},
        )
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.addFilter(_SemanticConsoleFilter(console_policy))
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    logging.captureWarnings(True)
    logging.getLogger("moabb").setLevel(logging.INFO)
    _console_policy = console_policy


def log_event(
    logger: logging.Logger,
    category: EventCategory,
    message: str,
    *,
    level: int = logging.INFO,
    epoch: int | None = None,
    total_epochs: int | None = None,
) -> None:
    """Emit one categorized record with optional epoch cadence metadata."""

    extra = {
        "event_category": category.value,
        "event_epoch": epoch,
        "event_total_epochs": total_epochs,
    }
    logger.log(level, message, extra=extra)
