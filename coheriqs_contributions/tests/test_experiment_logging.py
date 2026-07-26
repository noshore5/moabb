from dataclasses import dataclass
from enum import Enum
import json
import logging
import math
from pathlib import Path

from coheriqs_contributions.experiment_logging import (
    ConsolePolicy,
    EventCategory,
    configure_experiment_logging,
    log_event,
    serialize_event_data,
)


class _Mode(Enum):
    FAST = "fast"


@dataclass
class _Settings:
    output: Path
    modes: tuple[_Mode, ...]


class _NotSerializable:
    pass


def test_structured_event_categories_are_available_to_other_components() -> None:
    values = {category.value for category in EventCategory}
    assert {
        "research_note",
        "effective_config",
        "runtime_context",
        "fit_summary",
    } <= values


def test_structured_event_data_is_deterministic_and_kept_out_of_console(
    tmp_path, capsys
) -> None:
    log_path = tmp_path / "experiment.log"
    configure_experiment_logging(
        log_path, console_policy=ConsolePolicy(), overwrite=True
    )

    log_event(
        logging.getLogger("structured-test"),
        EventCategory.EFFECTIVE_CONFIG,
        "Configuration captured.",
        data={"zeta": 1, "run_id": "run-1", "schema_version": 1, "alpha": 2},
    )
    log_event(
        logging.getLogger("structured-test"),
        EventCategory.RUNTIME_CONTEXT,
        "Runtime captured.",
        data={"schema_version": 1, "run_id": "run-1"},
    )
    log_event(
        logging.getLogger("structured-test"),
        EventCategory.RESEARCH_NOTE,
        "Description hidden by default.",
    )

    assert capsys.readouterr().out == ""
    lines = log_path.read_text(encoding="utf-8").splitlines()
    line = lines[0]
    assert line.endswith(
        ' | data={"schema_version":1,"run_id":"run-1","alpha":2,"zeta":1}'
    )
    data_lines = [line.rsplit(" | data=", 1)[1] for line in lines if " | data=" in line]
    assert [json.loads(data_line) for data_line in data_lines][0]["alpha"] == 2


def test_structured_event_data_converts_common_values_and_marks_fallback() -> None:
    rendered = serialize_event_data(
        {
            "schema_version": 1,
            "config": _Settings(Path("result"), (_Mode.FAST,)),
            "unknown": _NotSerializable(),
        }
    )

    assert '"output":"result"' in rendered
    assert '"modes":["fast"]' in rendered
    assert "<non_json:" in rendered


def test_structured_event_data_is_strict_json_with_stable_nested_fallbacks() -> None:
    data = {
        "schema_version": 1,
        "nested": {"zeta": math.inf, "alpha": _NotSerializable()},
        "nan": math.nan,
    }

    first = serialize_event_data(data)
    second = serialize_event_data(data)

    assert first == second
    parsed = json.loads(first)
    assert parsed["nested"]["alpha"] == (
        "<non_json:test_experiment_logging._NotSerializable>"
    )
    assert parsed["nested"]["zeta"] == "<non_finite:inf>"
    assert parsed["nan"] == "<non_finite:nan>"
    assert first.index('"alpha"') < first.index('"zeta"')


def test_structured_event_data_escapes_unicode_line_separators() -> None:
    rendered = serialize_event_data({"description": "one\u0085two\u2028three\u2029four"})

    assert "\u0085" not in rendered
    assert "\u2028" not in rendered
    assert "\u2029" not in rendered
    assert "\\u0085" in rendered
    assert "\\u2028" in rendered
    assert "\\u2029" in rendered
    assert json.loads(rendered)["description"] == "one\u0085two\u2028three\u2029four"


def test_new_categories_follow_existing_console_detail_policies(tmp_path, capsys) -> None:
    log_path = tmp_path / "experiment.log"
    configure_experiment_logging(
        log_path,
        console_policy=ConsolePolicy(initial_details=True, final_results=False),
        overwrite=True,
    )
    logger = logging.getLogger("policy-test")
    log_event(logger, EventCategory.RESEARCH_NOTE, "Description visible.")
    log_event(logger, EventCategory.EFFECTIVE_CONFIG, "Configuration visible.")
    log_event(logger, EventCategory.RUNTIME_CONTEXT, "Runtime visible.")
    log_event(logger, EventCategory.FIT_SUMMARY, "Fit hidden.")

    output = capsys.readouterr().out
    assert "Description visible." in output
    assert "Configuration visible." in output
    assert "Runtime visible." in output
    assert "Fit hidden." not in output
