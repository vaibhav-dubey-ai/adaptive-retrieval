#!/usr/bin/env python
"""Run a command (and its children) under a hard RAM cap. Windows only.

Usage:  python scripts/run_capped.py [--fraction 0.9] -- <command> [args...]

The command and every process it spawns share one job object whose total
committed memory is limited to ``fraction`` of physical RAM; past that,
allocations fail inside the job instead of freezing the machine.
"""

import argparse
import ctypes
import subprocess
import sys
from ctypes import wintypes

JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
EXTENDED_LIMIT_INFO = 9


class _Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("r", "w", "o", "rb", "wb", "ob")]


class _Extended(ctypes.Structure):
    _fields_ = [("Basic", _Basic), ("Io", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.9)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given")

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    mem = _MemStatus(dwLength=ctypes.sizeof(_MemStatus))
    k32.GlobalMemoryStatusEx(ctypes.byref(mem))
    limit = int(mem.ullTotalPhys * args.fraction)

    job = k32.CreateJobObjectW(None, None)
    info = _Extended()
    info.Basic.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = limit
    if not k32.SetInformationJobObject(job, EXTENDED_LIMIT_INFO, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())

    print(f"[run_capped] limit {limit / 2**30:.2f} GiB of {mem.ullTotalPhys / 2**30:.2f} GiB RAM", file=sys.stderr)
    # no idle sleep while the command runs (screen lock is fine); cleared when this process exits.
    # shortcut: lid close / manual Sleep still suspend, no API can block those
    k32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    proc = subprocess.Popen(cmd)
    # shortcut: child starts a moment before it joins the job; fine for long-lived servers
    if not k32.AssignProcessToJobObject(job, int(proc._handle)):
        proc.kill()
        raise ctypes.WinError(ctypes.get_last_error())
    return proc.wait()


if __name__ == "__main__":
    sys.exit(main())
