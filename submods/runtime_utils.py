"""Shared runtime helpers for facetselfcal: terminal output, shell command execution,
resource-phase tracking, machine-info logging and the run-directory lock.

These live outside ``facetselfcal/main.py`` so that submodules can use them
without importing ``main``. ``main`` registers its logger and the active
resource monitor here via :func:`set_logger` and :func:`set_resource_monitor`.
"""
import builtins
import fcntl
import json
import logging
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

try:
    from facetselfcal.resource_chart import (DP3_EXECUTION_END_MARKER,
                                             DP3_EXECUTION_START_MARKER,
                                             dp3_command_phases,
                                             dp3_workflow_step_kinds)
except ImportError:
    from resource_chart import (DP3_EXECUTION_END_MARKER,
                                DP3_EXECUTION_START_MARKER,
                                dp3_command_phases,
                                dp3_workflow_step_kinds)

logger = logging.getLogger(__name__)
_RESOURCE_MONITOR = None


def set_logger(new_logger):
    """Set the logger used by :func:`run` (main.py passes its own logger)."""
    global logger
    logger = new_logger


def get_logger():
    """Return the logger registered with :func:`set_logger`."""
    return logger


def set_resource_monitor(monitor):
    """Register the active ResourceMonitor (or None) used by :func:`run`."""
    global _RESOURCE_MONITOR
    _RESOURCE_MONITOR = monitor


def get_resource_monitor():
    """Return the active ResourceMonitor, or None."""
    return _RESOURCE_MONITOR


def terminal_print(*args, **kwargs):
    """Print a message with the facetselfcal terminal prefix.

    Parameters
    ----------
    *args : object
        Message values to join with ``sep`` and print.
    **kwargs : object
        Keyword arguments forwarded to :func:`print`; ``sep`` controls the
        separator used to join message values.

    Returns
    -------
    None
    """
    if not args:
        builtins.print(**kwargs)
        return

    separator = kwargs.pop("sep", " ") or " "
    message = separator.join(str(arg) for arg in args)
    prefixed_message = "\n".join(
        f"[facetselfcal] {line}" for line in message.split("\n")
    )
    builtins.print(prefixed_message, **kwargs)


def run(command, log=False, taql=False):
    """
    Execute a shell command through subprocess

    Parameters
    ----------
    command : str
        Shell command to execute.
    log : bool, optional
        Print and log the command before running it.
    taql : bool, optional
        Capture and decode the command output using the TAQL execution path.

    Returns
    -------
    retval : int
        the return code of the executed process.
    """
    dp3_phases = dp3_command_phases(command)
    dp3_execution_id = (
        uuid.uuid4().hex if dp3_workflow_step_kinds(command) else None
    )
    if log:
        terminal_print('Command:', command)
    if log or dp3_phases:
        logger.info(command)
    if dp3_execution_id is not None:
        logger.info(
            DP3_EXECUTION_START_MARKER
            + json.dumps(
                {"id": dp3_execution_id, "command": command},
                separators=(",", ":"),
            )
        )
    resource_monitor = _RESOURCE_MONITOR
    dp3_start_epoch = (
        time.time()
        if resource_monitor is not None and dp3_phases
        else None
    )
    try:
        if taql:
            process = subprocess.run(
                command, shell=True, capture_output=True, encoding="utf-8"
            )
        else:
            process = subprocess.run(
                command, shell=True, stderr=subprocess.STDOUT, encoding="utf-8"
            )
        if dp3_execution_id is not None:
            logger.info(
                DP3_EXECUTION_END_MARKER
                + json.dumps(
                    {
                        "id": dp3_execution_id,
                        "returncode": process.returncode,
                    },
                    separators=(",", ":"),
                )
            )
    finally:
        if dp3_start_epoch is not None:
            resource_monitor.record_phase_intervals(
                dp3_phases, dp3_start_epoch, time.time()
            )
    retval = process.returncode
    #stdout = process.stdout
    stderr = process.stderr

    if taql: # to catch the error in taql which does not give a non-zero return value but instead writes "Error" in the output
        if "Error" in stderr:
            terminal_print("FAILED to run", command)
            terminal_print("stderr is", stderr)
            raise Exception(command)

    if retval!= 0:
        terminal_print("FAILED to run", command)
        terminal_print("return value is", retval)
        terminal_print("stderr is", stderr)
        #print("stdout is", stdout)
        raise Exception(command)
    return retval


_RUN_LOCK_FILE = ".facetselfcal.lock"


class _RunDirectoryBusyError(RuntimeError):
    pass


@contextmanager
def _run_directory_lock(run_directory):
    """Hold an exclusive, process-released lock for a run directory."""
    misc_directory = Path(run_directory) / "misc"
    misc_directory.mkdir(parents=True, exist_ok=True)
    lock_path = misc_directory / _RUN_LOCK_FILE

    with lock_path.open("a", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(
                lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
            )
        except BlockingIOError as exc:
            raise _RunDirectoryBusyError(
                "Another facetselfcal run is already active in {}.".format(
                    Path(run_directory).resolve()
                )
            ) from exc
        yield


def _track_resource_phase(phase):
    """Mark a workflow function's execution as a resource-monitor phase."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            monitor = get_resource_monitor()
            if monitor is None:
                return function(*args, **kwargs)

            previous_phase = monitor.set_phase(phase)
            try:
                return function(*args, **kwargs)
            finally:
                monitor.set_phase(previous_phase)

        return wrapped

    return decorate


@_track_resource_phase("predict")
def _run_predict_command(command):
    return run(command)


def _format_machine_bytes(byte_count):
    if byte_count is None:
        return "unavailable"
    return "{:.1f} GiB".format(byte_count / (1024 ** 3))


def _log_machine_info():
    try:
        hostname = socket.gethostname()
    except OSError:
        hostname = "unavailable"
    logger.info("Run host: %s", hostname)
    logger.info("Operating system: %s", platform.platform())

    logical_cpus = os.cpu_count()
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = None
    logger.info(
        "CPU count: %s logical; %s available to process",
        logical_cpus,
        available_cpus if available_cpus is not None else "unavailable",
    )

    ram_total = None
    ram_available = None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and total_pages > 0:
            ram_total = page_size * total_pages
    except (AttributeError, OSError, ValueError):
        pass
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        if page_size > 0 and available_pages >= 0:
            ram_available = page_size * available_pages
    except (AttributeError, OSError, ValueError):
        pass
    logger.info(
        "RAM: %s total; %s available",
        _format_machine_bytes(ram_total),
        _format_machine_bytes(ram_available),
    )

    run_directory = Path.cwd()
    try:
        disk_usage = shutil.disk_usage(run_directory)
    except OSError:
        logger.info("Disk at run directory %s: unavailable", run_directory)
    else:
        logger.info(
            "Disk at run directory %s: %s free of %s total",
            run_directory,
            _format_machine_bytes(disk_usage.free),
            _format_machine_bytes(disk_usage.total),
        )


def print_title(version):
    """
    Print the facetselfcal program title and version.

    Parameters
    ----------
    version : str
        Version string to display.

    Returns
    -------
    None
        The title is printed to standard output.
    """
    title = r"""
               _______    ___       ______  _______ .___________.
              |   ____|  /   \     /      ||   ____||           |
              |  |__    /  ^  \   |  ,----'|  |__   `---|  |----`
              |   __|  /  /_\  \  |  |     |   __|      |  |
              |  |    /  _____  \ |  `----.|  |____     |  |
              |__|   /__/     \__\ \______||_______|    |  |

         _______. _______  __       _______   ______      ___       __
        /       ||   ____||  |     |   ____| /      |    /   \     |  |
       |   (----`|  |__   |  |     |  |__   |  ,----'   /  ^  \    |  |
        \   \    |   __|  |  |     |   __|  |  |       /  /_\  \   |  |
    .----)   |   |  |____ |  `----.|  |     |  `----. /  _____  \  |  `----.
    |_______/    |_______||_______||__|      \______|/__/     \__\ |_______|


                                          Reinout van Weeren

                                 Reference: van Weeren et al.
                                      A&A, 651, A115 (2021)

                                  Facet-based self-calibration
                                             Version {version}

                                     Starting facetselfcal...
          """.format(version=version)

    lines = title.splitlines()
    artwork_line = max(
        (line for line in lines if line.strip() and line.strip()[0] in "_|/\\."),
        key=lambda line: len(line.strip()),
    )
    artwork_start = len(artwork_line) - len(artwork_line.lstrip())
    artwork_end = len(artwork_line.rstrip()) - 1
    artwork_center = (artwork_start + artwork_end) / 2
    centered_captions = {
        "Reinout van Weeren",
        "Reference: van Weeren et al.",
        "A&A, 651, A115 (2021)",
        "Facet-based self-calibration",
        f"Version {version}",
        "Starting facetselfcal...",
    }
    for index, line in enumerate(lines):
        caption = line.strip()
        if caption in centered_captions:
            padding = round(artwork_center - (len(caption) - 1) / 2)
            lines[index] = " " * padding + caption
    print("\n".join(lines))

    logger.info("VERSION: %s", version)
    return


def which(file_name):
    """
    Find an executable on ``PATH``.

    Parameters
    ----------
    file_name : str
        Executable filename to search for.

    Returns
    -------
    str or None
        Full executable path, or ``None`` when not found.
    """
    for path in os.environ["PATH"].split(os.pathsep):
        full_path = os.path.join(path, file_name)
        if os.path.exists(full_path) and os.access(full_path, os.X_OK):
            return full_path
    return None


def _with_run_directory_lock(run_function):
    """Prevent simultaneous workflows from sharing one working directory."""
    @wraps(run_function)
    def wrapped(*run_args, **run_kwargs):
        if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
            return run_function(*run_args, **run_kwargs)

        try:
            with _run_directory_lock(Path.cwd()):
                return run_function(*run_args, **run_kwargs)
        except _RunDirectoryBusyError as exc:
            terminal_print(str(exc))
            raise SystemExit(1) from None

    return wrapped
