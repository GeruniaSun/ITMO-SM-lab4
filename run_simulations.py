"""Run the scenarios listed in CSV, capture each result window, and log inputs.

Run from this project with ``python run_scenarios.py``. Edit
``scenario_ranges.csv`` to add or change scenarios. The source ``ms_lab4.alp``
is never modified.

This script uses only Python's standard library and Windows desktop APIs.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
from decimal import Decimal, InvalidOperation
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "ms_lab4.alp"
MODEL_NAME = "ms_lab4"
EXPERIMENT_NAME = "Simulation"
MODEL_WINDOW_TITLE = "ms_lab4 : Simulation"
SCENARIOS_FILE = PROJECT_DIR / "params.csv"

DEFAULT_ANYLOGIC = Path(
    r"C:\Program Files\AnyLogic 8.9 Personal Learning Edition\AnyLogic.exe"
)
ANYLOGIC_EXE = Path(os.environ.get("ANYLOGIC_EXE", str(DEFAULT_ANYLOGIC)))

RESULTS_DIR = PROJECT_DIR / "results"
SIMULATION_DAYS = 365
INFECTED_ON_START = 5
RANDOM_SEED = 1  # Keep the model's existing fixed seed for comparable runs.

MODEL_PARAMETERS = (
    "infectProb",
    "recoveryProb",
    "deathProb",
    "vaccineProb",
    "alpha",
    "vaccineTime",
    "n",
)
NETWORK_PARAMETERS = (
    "connectionsPerAgent",
    "connectionsRange",
    "neighborLinkFraction",
    "m",
)
PARAMETERS = (*MODEL_PARAMETERS, *NETWORK_PARAMETERS)
PROBABILITY_PARAMETERS = (
    "infectProb",
    "recoveryProb",
    "deathProb",
    "vaccineProb",
    "alpha",
    "neighborLinkFraction",
)
INTEGER_PARAMETERS = ("vaccineTime", "n", "connectionsPerAgent", "m")

# These values are read from Main's EnvironmentProperties in ms_lab4.alp.
NETWORK_PARAMETER_DEFAULTS: dict[str, int | float] = {
    "connectionsPerAgent": 10,
    "connectionsRange": 50,
    "neighborLinkFraction": 0.95,
    "m": 10,
}

# Allowed AnyLogic network types. Each scenario row selects one of these.
NETWORK_TYPES = (
    "RANDOM",
    "SMALL_WORLD",
    "RING_LATTICE",
    "SCALE_FREE",
    "ALL_IN_RANGE",
)
NETWORK_ACTIVE_PARAMETERS = {
    "RANDOM": ("connectionsPerAgent",),
    "SMALL_WORLD": ("connectionsPerAgent", "neighborLinkFraction"),
    "RING_LATTICE": ("connectionsPerAgent",),
    "SCALE_FREE": ("m",),
    "ALL_IN_RANGE": ("connectionsRange",),
}

SCENARIO_CSV_COLUMNS = (*MODEL_PARAMETERS, "networkType", *NETWORK_PARAMETERS)
CSV_COLUMNS = (
    "run_id",
    "networkType",
    *PARAMETERS,
    "infectedOnStart",
    "randomSeed",
    "screenshot",
)
LEGACY_CSV_COLUMNS = (
    "run_id",
    *MODEL_PARAMETERS,
    "infectedOnStart",
    "randomSeed",
    "screenshot",
)

WINDOW_TIMEOUT_SECONDS = 180
SIMULATION_TIMEOUT_SECONDS = 300
MINIMUM_RUN_SECONDS = 10
QUIET_CPU_SECONDS_PER_SECOND = 0.05
QUIET_SAMPLES_REQUIRED = 5
CPU_POLL_SECONDS = 1
TEMP_CLEANUP_ATTEMPTS = 12
TEMP_CLEANUP_RETRY_SECONDS = 0.75


def _format_number(value: int | float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Parameter value must be numeric, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"Parameter value must be finite, got {value!r}")
    if isinstance(value, int):
        return str(value)
    return format(value, ".15g")


def _parse_scenario_value(parameter: str, raw_value: str, line_number: int) -> int | float:
    try:
        value = Decimal(raw_value)
    except InvalidOperation as error:
        raise ValueError(
            f"Line {line_number}: {parameter!r} must be numeric, got {raw_value!r}."
        ) from error
    if not value.is_finite():
        raise ValueError(f"Line {line_number}: {parameter!r} must be a finite number.")
    if parameter in INTEGER_PARAMETERS:
        if value != value.to_integral_value():
            raise ValueError(f"Line {line_number}: {parameter!r} must be a whole number.")
        return int(value)
    return float(value)


def read_scenarios_csv(path: Path) -> list[dict[str, int | float | str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Scenario CSV not found: {path}")

    scenarios: list[dict[str, int | float | str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if tuple((name or "").strip() for name in (reader.fieldnames or ())) != SCENARIO_CSV_COLUMNS:
            raise ValueError(
                f"{path} must have exactly these columns, in this order: {', '.join(SCENARIO_CSV_COLUMNS)}."
            )

        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"Line {line_number} of {path} has more values than columns.")
            scenario: dict[str, int | float | str] = {}
            for parameter in MODEL_PARAMETERS:
                raw_value = (row.get(parameter) or "").strip()
                if not raw_value:
                    raise ValueError(f"Line {line_number} of {path} is missing {parameter!r}.")
                scenario[parameter] = _parse_scenario_value(parameter, raw_value, line_number)

            network_type = (row.get("networkType") or "").strip().upper()
            if not network_type:
                raise ValueError(f"Line {line_number} of {path} is missing 'networkType'.")
            scenario["networkType"] = network_type

            for parameter in NETWORK_PARAMETERS:
                raw_value = (row.get(parameter) or "").strip()
                if not raw_value:
                    raise ValueError(f"Line {line_number} of {path} is missing {parameter!r}.")
                scenario[parameter] = _parse_scenario_value(parameter, raw_value, line_number)
            scenarios.append(scenario)

    return validate_scenarios(scenarios)


def validate_scenarios(
    scenarios: Iterable[dict[str, int | float | str]],
) -> list[dict[str, int | float | str]]:
    validated = list(scenarios)
    if not validated:
        raise ValueError("No scenario rows were found in the CSV.")

    for index, scenario in enumerate(validated, start=1):
        required = {*PARAMETERS, "networkType"}
        missing = required - set(scenario)
        extra = set(scenario) - required
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"unknown: {', '.join(sorted(extra))}")
            raise ValueError(f"Scenario {index}: " + "; ".join(details))

        for name in PARAMETERS:
            value = scenario[name]
            _format_number(value)
            if name in PROBABILITY_PARAMETERS:
                if not 0 <= float(value) <= 1:
                    raise ValueError(f"Scenario {index}: {name} must be between 0 and 1.")
            elif name in INTEGER_PARAMETERS and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"Scenario {index}: {name} must be a positive integer.")
            elif name == "connectionsRange" and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            ):
                raise ValueError(f"Scenario {index}: {name} must be a positive number.")

        network_type = scenario["networkType"]
        if network_type not in NETWORK_TYPES or network_type not in NETWORK_ACTIVE_PARAMETERS:
            raise ValueError(f"Scenario {index}: unsupported networkType {network_type!r}.")
        connections = scenario["connectionsPerAgent"]
        if connections >= 500:
            raise ValueError(f"Scenario {index}: connectionsPerAgent must be less than 500.")
        if network_type in {"RING_LATTICE", "SMALL_WORLD"} and connections % 2 != 0:
            raise ValueError(
                f"Scenario {index}: connectionsPerAgent must be even for {network_type}."
            )
        if scenario["m"] >= 500:
            raise ValueError(f"Scenario {index}: m must be less than the 500-agent population.")
    return validated


def _replace_cdata_or_text(block: str, tag: str, value: str, *, count: int = 1) -> str:
    pattern = re.compile(rf"(<{re.escape(tag)}\b[^>]*>)(.*?)(</{re.escape(tag)}>)", re.DOTALL)
    matches = list(pattern.finditer(block))
    if len(matches) != count:
        raise ValueError(f"Expected {count} <{tag}> element(s), found {len(matches)}.")

    for match in reversed(matches):
        inner = match.group(2)
        if "<![CDATA[" in inner:
            cdata_pattern = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)
            if not cdata_pattern.search(inner):
                raise ValueError(f"Could not replace CDATA value in <{tag}>.")
            new_inner = cdata_pattern.sub(lambda _: f"<![CDATA[{value}]]>", inner, count=1)
        else:
            new_inner = value
        block = block[: match.start(2)] + new_inner + block[match.end(2) :]
    return block


def _get_named_parameter_blocks(xml_text: str) -> list[re.Match[str]]:
    pattern = re.compile(r"<Variable\b[^>]*\bClass=\"Parameter\"[^>]*>.*?</Variable>", re.DOTALL)
    return list(pattern.finditer(xml_text))


def _parameter_name(block: str) -> str | None:
    match = re.search(r"<Name>\s*<!\[CDATA\[(.*?)\]\]>\s*</Name>", block, re.DOTALL)
    return match.group(1) if match else None


def _read_parameter_default(block: str, parameter_name: str) -> str:
    default_match = re.search(
        r"<DefaultValue\b[^>]*>\s*<Code>\s*<!\[CDATA\[(.*?)\]\]>\s*</Code>\s*</DefaultValue>",
        block,
        re.DOTALL,
    )
    if not default_match:
        raise ValueError(f"Parameter {parameter_name!r} has no CodeValue DefaultValue.")
    return default_match.group(1).strip()


def replace_parameter_default(xml_text: str, parameter_name: str, value: str) -> str:
    blocks = _get_named_parameter_blocks(xml_text)
    matches = [match for match in blocks if _parameter_name(match.group(0)) == parameter_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one model parameter {parameter_name!r}, found {len(matches)}.")

    match = matches[0]
    block = match.group(0)
    pattern = re.compile(
        r"(<DefaultValue\b[^>]*>\s*<Code>\s*<!\[CDATA\[)(.*?)(\]\]>\s*</Code>\s*</DefaultValue>)",
        re.DOTALL,
    )
    updated, replacements = pattern.subn(lambda m: m.group(1) + value + m.group(3), block, count=1)
    if replacements != 1:
        raise ValueError(f"Could not update default value for {parameter_name!r}.")
    return xml_text[: match.start()] + updated + xml_text[match.end() :]


def _replace_single_named_block(
    xml_text: str,
    tag: str,
    name: str,
    updater,
) -> str:
    pattern = re.compile(rf"<{re.escape(tag)}\b[^>]*>.*?</{re.escape(tag)}>", re.DOTALL)
    matches: list[re.Match[str]] = []
    for match in pattern.finditer(xml_text):
        block_name = re.search(r"<Name>\s*<!\[CDATA\[(.*?)\]\]>\s*</Name>", match.group(0), re.DOTALL)
        if block_name and block_name.group(1) == name:
            matches.append(match)
    if len(matches) != 1:
        raise ValueError(f"Expected one <{tag}> named {name!r}, found {len(matches)}.")
    match = matches[0]
    updated = updater(match.group(0))
    return xml_text[: match.start()] + updated + xml_text[match.end() :]


def _set_model_time_properties(block: str) -> str:
    pattern = re.compile(r"<ModelTimeProperties\b[^>]*>.*?</ModelTimeProperties>", re.DOTALL)
    matches = list(pattern.finditer(block))
    if len(matches) != 1:
        raise ValueError(f"Expected one ModelTimeProperties block, found {len(matches)}.")
    match = matches[0]
    time_props = match.group(0)
    time_props = _replace_cdata_or_text(time_props, "StopOption", "Stop at specified time")
    time_props = _replace_cdata_or_text(time_props, "FinalTime", str(SIMULATION_DAYS))
    return block[: match.start()] + time_props + block[match.end() :]


def patch_model_xml(xml_text: str, scenario: dict[str, int | float | str]) -> str:
    for name in MODEL_PARAMETERS:
        xml_text = replace_parameter_default(xml_text, name, _format_number(scenario[name]))

    xml_text = _patch_main_network(xml_text, scenario)

    xml_text = _replace_single_named_block(
        xml_text,
        "RunConfiguration",
        "RunConfiguration",
        lambda block: _patch_run_configuration(block),
    )
    xml_text = _replace_single_named_block(
        xml_text,
        "SimulationExperiment",
        EXPERIMENT_NAME,
        lambda block: _patch_simulation_experiment(block),
    )

    # Validate the resulting XML without serializing it, preserving AnyLogic's CDATA and layout.
    ET.fromstring(xml_text)
    for name in MODEL_PARAMETERS:
        matching = [match.group(0) for match in _get_named_parameter_blocks(xml_text) if _parameter_name(match.group(0)) == name]
        if len(matching) != 1 or _read_parameter_default(matching[0], name) != _format_number(scenario[name]):
            raise ValueError(f"Post-edit validation failed for parameter {name!r}.")

    environment_matches = list(re.finditer(r"<EnvironmentProperties\b[^>]*>.*?</EnvironmentProperties>", xml_text, re.DOTALL))
    if not environment_matches:
        raise ValueError("Could not validate Main network settings: EnvironmentProperties was not found.")
    main_environment = environment_matches[0].group(0)
    expected_network_values = {
        "NetworkType": str(scenario["networkType"]),
        "ConnectionsPerAgentCode": _format_number(scenario["connectionsPerAgent"]),
        "ConnectionsRangeCode": _format_number(scenario["connectionsRange"]),
        "NeighborLinkFractionCode": _format_number(scenario["neighborLinkFraction"]),
        "MCode": _format_number(scenario["m"]),
    }
    for tag, expected in expected_network_values.items():
        value_match = re.search(rf"<{re.escape(tag)}\b[^>]*>(.*?)</{re.escape(tag)}>", main_environment, re.DOTALL)
        if not value_match:
            raise ValueError(f"Could not validate Main network setting {tag!r}.")
        actual = re.sub(r"^\s*<!\[CDATA\[(.*?)\]\]>\s*$", r"\1", value_match.group(1), flags=re.DOTALL).strip()
        if actual != expected:
            raise ValueError(f"Post-edit validation failed for network setting {tag!r}.")

    infected_blocks = [
        match.group(0)
        for match in _get_named_parameter_blocks(xml_text)
        if _parameter_name(match.group(0)) == "infectedOnStart"
    ]
    if len(infected_blocks) != 1 or _read_parameter_default(infected_blocks[0], "infectedOnStart") != str(INFECTED_ON_START):
        raise ValueError("infectedOnStart changed or could not be verified as 5.")
    return xml_text


def _patch_main_network(xml_text: str, scenario: dict[str, int | float | str]) -> str:
    # Main is the first ActiveObjectClass in this .alp; its environment settings
    # define the social graph used by the 500-agent humans population.
    pattern = re.compile(r"<EnvironmentProperties\b[^>]*>.*?</EnvironmentProperties>", re.DOTALL)
    match = pattern.search(xml_text)
    if not match:
        raise ValueError("Could not find Main's EnvironmentProperties network configuration.")

    block = match.group(0)
    block = _replace_cdata_or_text(block, "NetworkType", str(scenario["networkType"]))
    network_tags = {
        "connectionsPerAgent": "ConnectionsPerAgentCode",
        "connectionsRange": "ConnectionsRangeCode",
        "neighborLinkFraction": "NeighborLinkFractionCode",
        "m": "MCode",
    }
    for parameter, tag in network_tags.items():
        block = _replace_cdata_or_text(block, tag, _format_number(scenario[parameter]))
    return xml_text[: match.start()] + block + xml_text[match.end() :]


def _patch_run_configuration(block: str) -> str:
    block = _set_model_time_properties(block)
    pattern = re.compile(r"<AnimationProperties\b[^>]*>.*?</AnimationProperties>", re.DOTALL)
    matches = list(pattern.finditer(block))
    if len(matches) != 1:
        raise ValueError(f"Expected one AnimationProperties block, found {len(matches)}.")
    match = matches[0]
    animation = match.group(0)
    animation = _replace_cdata_or_text(animation, "ExecutionMode", "virtualTime")
    animation = _replace_cdata_or_text(animation, "StopNever", "false")
    return block[: match.start()] + animation + block[match.end() :]


def _patch_simulation_experiment(block: str) -> str:
    block = _set_model_time_properties(block)
    pattern = re.compile(r"<PresentationProperties\b[^>]*>.*?</PresentationProperties>", re.DOTALL)
    matches = list(pattern.finditer(block))
    if len(matches) != 1:
        raise ValueError(f"Expected one PresentationProperties block, found {len(matches)}.")
    match = matches[0]
    presentation = _replace_cdata_or_text(match.group(0), "ExecutionMode", "virtualTime")
    return block[: match.start()] + presentation + block[match.end() :]


def read_text_preserving_newlines(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as file:
        return file.read()


def write_text_preserving_newlines(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(text)


def _tasklist_processes() -> list[tuple[str, int]]:
    result = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    processes: list[tuple[str, int]] = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) < 2:
            continue
        try:
            processes.append((row[0], int(row[1].replace(",", ""))))
        except ValueError:
            continue
    return processes


def _windows_api():
    if os.name != "nt":
        raise RuntimeError("This script requires Windows to launch and capture the AnyLogic desktop window.")

    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = ctypes.c_void_p
    user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
    user32.ReleaseDC.restype = ctypes.c_int
    user32.EnumWindows.restype = wintypes.BOOL

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.GetProcessTimes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL

    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.SelectObject.restype = ctypes.c_void_p
    gdi32.BitBlt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.DWORD,
    ]
    gdi32.BitBlt.restype = wintypes.BOOL
    gdi32.GetDIBits.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.UINT,
        wintypes.UINT,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.UINT,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    gdi32.DeleteObject.restype = wintypes.BOOL
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
    gdi32.DeleteDC.restype = wintypes.BOOL

    try:
        user32.SetProcessDPIAware()
    except AttributeError:
        pass
    return user32, kernel32, gdi32, wintypes


def _visible_windows(user32, wintypes) -> list[tuple[int, str, int]]:
    windows: list[tuple[int, str, int]] = []
    enum_proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        windows.append((int(hwnd), title, int(pid.value)))
        return True

    callback_ref = enum_proc_type(callback)
    user32.EnumWindows.argtypes = [enum_proc_type, wintypes.LPARAM]
    if not user32.EnumWindows(callback_ref, 0):
        error = ctypes.get_last_error()
        if error:
            raise OSError(error, "EnumWindows failed")
    return windows


def _is_model_window_title(title: str) -> bool:
    # Match AnyLogic's model window title as a phrase. Searching for the words
    # "ms_lab4" and "simulation" separately also matches this script's name in
    # IDE window titles such as "run_simulations.py - ms_lab4 - Visual Studio Code".
    return MODEL_WINDOW_TITLE.casefold() in title.casefold()


def _matching_windows(user32, wintypes) -> list[tuple[int, str, int]]:
    return [
        window
        for window in _visible_windows(user32, wintypes)
        if _is_model_window_title(window[1])
    ]


def find_model_window(user32, wintypes) -> tuple[int, str, int] | None:
    matches = _matching_windows(user32, wintypes)
    if not matches:
        return None
    exact = [item for item in matches if MODEL_WINDOW_TITLE.casefold() in item[1].casefold()]
    return exact[0] if exact else matches[0]


def wait_for_model_window(user32, wintypes, process: subprocess.Popen, baseline_pids: set[int]):
    deadline = time.monotonic() + WINDOW_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        window = find_model_window(user32, wintypes)
        if window:
            hwnd, title, window_pid = window
            if window_pid in baseline_pids:
                raise RuntimeError(
                    "The model opened inside an AnyLogic process that was already running. "
                    "Close AnyLogic before starting the batch so the script can safely stop "
                    "only processes it launched."
                )
            return hwnd, title, window_pid
        if process.poll() is not None and process.returncode not in (0, None):
            raise RuntimeError(f"AnyLogic exited with code {process.returncode} before opening the model window.")
        time.sleep(0.5)
    raise TimeoutError(
        f"Timed out after {WINDOW_TIMEOUT_SECONDS}s waiting for a window containing "
        f"{MODEL_NAME!r} and {EXPERIMENT_NAME!r}. Check AnyLogic startup and licensing dialogs."
    )


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


def _filetime_seconds(value: FILETIME) -> float:
    return ((value.dwHighDateTime << 32) | value.dwLowDateTime) / 10_000_000


def process_cpu_seconds(kernel32, pid: int) -> float:
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"Could not inspect AnyLogic process {pid}.")
    try:
        creation = FILETIME()
        exit_time = FILETIME()
        kernel_time = FILETIME()
        user_time = FILETIME()
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise OSError(ctypes.get_last_error(), f"Could not read CPU time for process {pid}.")
        return _filetime_seconds(kernel_time) + _filetime_seconds(user_time)
    finally:
        kernel32.CloseHandle(handle)


def wait_for_simulation_completion(kernel32, pid: int) -> None:
    """Wait until the virtual-time model's owning process is idle after startup.

    The model copy has a finite 365-day stop time. In virtual-time mode AnyLogic
    advances without wall-clock pacing; its model engine consumes CPU while
    advancing and becomes idle after the stop event. Require several consecutive
    quiet samples before taking the screenshot.
    """
    start = time.monotonic()
    time.sleep(MINIMUM_RUN_SECONDS)
    previous_cpu = process_cpu_seconds(kernel32, pid)
    quiet_samples = 0

    while time.monotonic() - start < SIMULATION_TIMEOUT_SECONDS:
        time.sleep(CPU_POLL_SECONDS)
        current_cpu = process_cpu_seconds(kernel32, pid)
        cpu_delta = current_cpu - previous_cpu
        previous_cpu = current_cpu
        if cpu_delta <= QUIET_CPU_SECONDS_PER_SECOND:
            quiet_samples += 1
            if quiet_samples >= QUIET_SAMPLES_REQUIRED:
                return
        else:
            quiet_samples = 0

    raise TimeoutError(
        f"AnyLogic did not become idle within {SIMULATION_TIMEOUT_SECONDS}s; "
        "no screenshot was recorded for this run."
    )


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class RGBQUAD(ctypes.Structure):
    _fields_ = [
        ("rgbBlue", ctypes.c_uint8),
        ("rgbGreen", ctypes.c_uint8),
        ("rgbRed", ctypes.c_uint8),
        ("rgbReserved", ctypes.c_uint8),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", RGBQUAD * 1)]


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    payload = chunk_type + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def capture_window_png(user32, gdi32, wintypes, hwnd: int, destination: Path) -> None:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise OSError(ctypes.get_last_error(), "Could not determine the model window bounds.")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width < 32 or height < 32:
        raise RuntimeError(f"The model window has invalid dimensions: {width}x{height}.")

    SW_RESTORE = 9
    SRCCOPY = 0x00CC0020
    CAPTUREBLT = 0x40000000
    DIB_RGB_COLORS = 0
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.5)

    screen_dc = user32.GetDC(None)
    if not screen_dc:
        raise OSError(ctypes.get_last_error(), "Could not access the desktop for window capture.")
    memory_dc = None
    bitmap = None
    previous_object = None
    try:
        memory_dc = gdi32.CreateCompatibleDC(screen_dc)
        bitmap = gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        if not memory_dc or not bitmap:
            raise OSError(ctypes.get_last_error(), "Could not allocate a bitmap for the screenshot.")
        previous_object = gdi32.SelectObject(memory_dc, bitmap)
        if not previous_object:
            raise OSError(ctypes.get_last_error(), "Could not select the screenshot bitmap.")
        if not gdi32.BitBlt(
            memory_dc,
            0,
            0,
            width,
            height,
            screen_dc,
            rect.left,
            rect.top,
            SRCCOPY | CAPTUREBLT,
        ):
            raise OSError(ctypes.get_last_error(), "Could not copy the model window to the screenshot bitmap.")
        gdi32.SelectObject(memory_dc, previous_object)

        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height  # Top-down rows.
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = 0
        pixels = ctypes.create_string_buffer(width * height * 4)
        lines = gdi32.GetDIBits(
            screen_dc,
            bitmap,
            0,
            height,
            ctypes.cast(pixels, ctypes.c_void_p),
            ctypes.byref(info),
            DIB_RGB_COLORS,
        )
        if lines != height:
            raise OSError(ctypes.get_last_error(), f"Screenshot returned {lines} of {height} rows.")

        bgra = pixels.raw
        scanlines = bytearray()
        bytes_per_row = width * 4
        for row in range(height):
            scanlines.append(0)  # PNG filter: None.
            row_start = row * bytes_per_row
            row_end = row_start + bytes_per_row
            raw_row = bgra[row_start:row_end]
            for pixel in range(0, bytes_per_row, 4):
                scanlines.extend((raw_row[pixel + 2], raw_row[pixel + 1], raw_row[pixel]))

        png = bytearray(b"\x89PNG\r\n\x1a\n")
        png.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        png.extend(_png_chunk(b"IDAT", zlib.compress(bytes(scanlines), level=6)))
        png.extend(_png_chunk(b"IEND", b""))
        destination.write_bytes(png)
    finally:
        if memory_dc and previous_object:
            gdi32.SelectObject(memory_dc, previous_object)
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory_dc:
            gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(None, screen_dc)


def terminate_launched_processes(process_ids: Iterable[int], baseline_pids: set[int]) -> None:
    targets = [pid for pid in dict.fromkeys(process_ids) if pid > 0 and pid not in baseline_pids]
    for pid in targets:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    if not targets:
        return

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        running = {pid for _image, pid in _tasklist_processes()}
        survivors = [pid for pid in targets if pid in running]
        if not survivors:
            return
        time.sleep(0.25)
    raise RuntimeError(f"Could not confirm termination of AnyLogic process(es): {survivors}.")


def migrate_legacy_manifest(manifest_path: Path) -> None:
    if not manifest_path.exists():
        return

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = tuple(reader.fieldnames or ())
        if columns == CSV_COLUMNS:
            return
        if columns != LEGACY_CSV_COLUMNS:
            raise ValueError(
                f"{manifest_path} has an unsupported CSV schema. Expected the current or previous runs.csv format."
            )
        old_rows = list(reader)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=manifest_path.parent,
            prefix="runs_migration_",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for old_row in old_rows:
                writer.writerow({
                    "run_id": old_row["run_id"],
                    "networkType": "RANDOM",
                    **{name: old_row[name] for name in MODEL_PARAMETERS},
                    **NETWORK_PARAMETER_DEFAULTS,
                    "infectedOnStart": old_row["infectedOnStart"],
                    "randomSeed": old_row["randomSeed"],
                    "screenshot": old_row["screenshot"],
                })
        os.replace(temporary_path, manifest_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def next_run_id(results_dir: Path, manifest_path: Path) -> int:
    migrate_legacy_manifest(manifest_path)
    highest = 0
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames and tuple(reader.fieldnames) != CSV_COLUMNS:
                raise ValueError(
                    f"{manifest_path} has a different CSV schema. Move it aside or use a new results directory."
                )
            for row in reader:
                try:
                    highest = max(highest, int(row["run_id"]))
                except (TypeError, ValueError, KeyError):
                    continue
    while (results_dir / f"run_{highest + 1:03d}.png").exists():
        highest += 1
    return highest + 1


def append_manifest(manifest_path: Path, row: dict[str, str | int | float]) -> None:
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0
    with manifest_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def make_model_copy(
    scenario: dict[str, int | float | str],
    run_id: int,
) -> tuple[tempfile.TemporaryDirectory, Path]:
    temporary = tempfile.TemporaryDirectory(
        prefix=f"anylogic_run_{run_id:03d}_",
        ignore_cleanup_errors=True,
    )
    run_dir = Path(temporary.name)
    try:
        model_copy = run_dir / MODEL_FILE.name
        shutil.copy2(MODEL_FILE, model_copy)
        images_dir = PROJECT_DIR / "images"
        if images_dir.is_dir():
            shutil.copytree(images_dir, run_dir / "images")

        xml_text = read_text_preserving_newlines(model_copy)
        patched = patch_model_xml(xml_text, scenario)
        write_text_preserving_newlines(model_copy, patched)
        return temporary, model_copy
    except Exception:
        cleanup_temporary_directory(temporary)
        raise


def cleanup_temporary_directory(temporary: tempfile.TemporaryDirectory) -> bool:
    """Retry Windows temp removal after AnyLogic exits and releases open files."""
    temp_path = Path(temporary.name)
    for attempt in range(TEMP_CLEANUP_ATTEMPTS):
        try:
            temporary.cleanup()
        except OSError:
            # Windows can briefly keep a handle open after taskkill returns.
            pass
        if not temp_path.exists():
            return True
        if attempt + 1 < TEMP_CLEANUP_ATTEMPTS:
            time.sleep(TEMP_CLEANUP_RETRY_SECONDS)

    print(
        f"Warning: could not remove temporary model folder yet; leaving it for later cleanup: {temp_path}",
        file=sys.stderr,
    )
    return False


def run_scenario(
    run_id: int,
    scenario: dict[str, int | float | str],
    screenshot_path: Path,
    anylogic_exe: Path,
    user32,
    kernel32,
    gdi32,
    wintypes,
) -> None:
    baseline_pids = {pid for _image, pid in _tasklist_processes()}
    preexisting_windows = _visible_windows(user32, wintypes)
    anylogic_windows = [window for window in preexisting_windows if "anylogic" in window[1].casefold()]
    existing_model_windows = [
        window
        for window in preexisting_windows
        if _is_model_window_title(window[1])
    ]
    if anylogic_windows or existing_model_windows:
        detected_titles = sorted({window[1] for window in anylogic_windows + existing_model_windows})
        detected = f" Detected window(s): {', '.join(detected_titles)}." if detected_titles else ""
        raise RuntimeError(
            "Close all existing AnyLogic windows before running this script. "
            "The script will not risk terminating an AnyLogic process it did not launch."
            + detected
        )

    temporary, model_copy = make_model_copy(scenario, run_id)
    process: subprocess.Popen | None = None
    window_pid = 0
    try:
        print(f"[run {run_id:03d}] Launching {EXPERIMENT_NAME} for {SIMULATION_DAYS} model days...")
        process = subprocess.Popen(
            [str(anylogic_exe), "-r", str(model_copy), EXPERIMENT_NAME],
            cwd=str(anylogic_exe.parent),
        )
        hwnd, title, window_pid = wait_for_model_window(user32, wintypes, process, baseline_pids)
        print(f"[run {run_id:03d}] Window ready: {title} (PID {window_pid}). Waiting for completion...")
        wait_for_simulation_completion(kernel32, window_pid)
        capture_window_png(user32, gdi32, wintypes, hwnd, screenshot_path)
        print(f"[run {run_id:03d}] Screenshot saved: {screenshot_path}")
    finally:
        target_pids = [window_pid]
        if process is not None:
            target_pids.append(process.pid)
        cleanup_error = None
        try:
            terminate_launched_processes(target_pids, baseline_pids)
        except Exception as error:
            cleanup_error = error
        finally:
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if process.pid not in baseline_pids:
                        process.kill()
                        process.wait(timeout=5)
            cleanup_temporary_directory(temporary)
        if cleanup_error is not None:
            raise cleanup_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anylogic",
        type=Path,
        default=ANYLOGIC_EXE,
        help="Path to AnyLogic.exe (default: installed AnyLogic 8.9 PLE path or ANYLOGIC_EXE env var).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for PNG screenshots and runs.csv (default: ./results).",
    )
    parser.add_argument(
        "--scenario-ranges",
        type=Path,
        default=SCENARIOS_FILE,
        help="CSV with one parameter configuration per row (default: ./scenario_ranges.csv).",
    )
    return parser.parse_args()


def main() -> int:
    if os.name != "nt":
        print("This script requires Windows.", file=sys.stderr)
        return 2
    args = parse_args()
    try:
        scenarios = read_scenarios_csv(args.scenario_ranges.resolve())
    except Exception as error:
        print(f"Scenario configuration error: {error}", file=sys.stderr)
        return 2
    anylogic_exe = args.anylogic.resolve()
    if not MODEL_FILE.is_file():
        print(f"Model file not found: {MODEL_FILE}", file=sys.stderr)
        return 2
    if not anylogic_exe.is_file():
        print(
            f"AnyLogic.exe not found: {anylogic_exe}\n"
            "Pass --anylogic <path> or set the ANYLOGIC_EXE environment variable.",
            file=sys.stderr,
        )
        return 2

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "runs.csv"
    try:
        run_id = next_run_id(args.output_dir, manifest_path)
        print(f"Loaded {len(scenarios)} explicit scenarios from {args.scenario_ranges.resolve()}.")
        user32, kernel32, gdi32, wintypes = _windows_api()
        for scenario in scenarios:
            screenshot_path = args.output_dir / f"run_{run_id:03d}.png"
            run_scenario(
                run_id,
                scenario,
                screenshot_path,
                anylogic_exe,
                user32,
                kernel32,
                gdi32,
                wintypes,
            )
            relative_screenshot = os.path.relpath(screenshot_path, args.output_dir)
            row: dict[str, str | int | float] = {
                "run_id": run_id,
                "networkType": scenario["networkType"],
                **scenario,
                "infectedOnStart": INFECTED_ON_START,
                "randomSeed": RANDOM_SEED,
                "screenshot": relative_screenshot,
            }
            append_manifest(manifest_path, row)
            run_id += 1
        print(f"Completed {len(scenarios)} runs. Manifest: {manifest_path}")
        return 0
    except Exception as error:
        print(f"Run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
