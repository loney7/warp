# Unified Logging

**Issues**: [GH-1315](https://github.com/NVIDIA/warp/issues/1315) (host-side logging infrastructure), [GH-1194](https://github.com/NVIDIA/warp/issues/1194) (kernel-side `wp.log()`)

## Motivation

Warp needs logging on both sides of the Python/kernel boundary:

1. **Host-side diagnostics.** Warp itself emits compilation messages, deprecation
   warnings, and errors during module loading, kernel compilation, and array
   operations. The host-side logging infrastructure routes these through a single
   pluggable interface controlled by `wp.config.log_level`.

2. **Kernel-side user logging.** Users debugging GPU kernels need `printf`-style
   visibility into per-thread state. A `wp.log()` builtin writes to a ring buffer
   on the device and drains records to the host on synchronization.

This document defines the shared conventions for both so that they feel like a
single, coherent feature rather than two unrelated systems.

## Requirements

| ID  | Requirement                                                                 | Priority | Notes                                         |
| --- | --------------------------------------------------------------------------- | -------- | --------------------------------------------- |
| R1  | Single set of log-level constants shared by host and kernel code            | Must     |                                               |
| R2  | Kernel-side `wp.log_*` builtins for printf-style logging from kernels       | Must     | Host-side emitters are internal-only          |
| R3  | One user-facing knob (`config.log_level`) controls the default threshold    | Must     | Power users can override per-logger            |
| R4  | Host-side warnings integrate with Python's `warnings` filter machinery      | Must     | `-W` flags, `simplefilter()`, etc.            |
| R5  | Kernel log records route to stdlib `logging` for per-module configuration   | Must     | `logging.getLogger("warp.kernel.{module}")`   |
| R6  | Pluggable host logger for framework integration (Omniverse Kit, etc.)       | Should   | `wp.Logger` Protocol + `wp.set_logger()`      |

**Non-goals:**

- Runtime-variable log levels inside kernels. The level is always a compile-time
  constant because message metadata is resolved at codegen time to avoid
  GPU-to-host string copies.
- Structured/machine-readable log output. The focus is human-readable diagnostics.
  Kernel log records carry an optional numeric payload (up to 128 bits) but not
  arbitrary structured data.

## Design

### Log-Level Constants

Four levels, defined once in `warp/_src/logger.py` and re-exported from
`warp/__init__.py`. Numeric values match the Python `logging` module:

| Constant         | Value | Purpose                                  |
| ---------------- | ----- | ---------------------------------------- |
| `wp.LOG_DEBUG`   | 10    | Verbose compilation/debugging output     |
| `wp.LOG_INFO`    | 20    | Informational (default threshold)        |
| `wp.LOG_WARNING` | 30    | Warnings and deprecation notices         |
| `wp.LOG_ERROR`   | 40    | Errors                                   |

All other modules (including `context.py` for kernel-side codegen) import from
`logger.py`. There is exactly one place where the values are defined.

### API Surface

The four `wp.log_*` names are kernel built-ins only. They are not exposed as
Python functions at host scope.

```python
# Kernel-side (inside @wp.kernel or @wp.func)
wp.log_debug("entering branch")
wp.log_info("iteration count %d", i)
wp.log_warning("unexpected value %f", x)
wp.log_error("constraint violated")
```

Host-side, Warp emits its own diagnostics through an internal module
(`warp/_src/logger.py`). Users do not call these emitters directly. To receive
Warp's host messages, frameworks register a custom logger via `wp.set_logger()`
(see "Output Routing" below).

The public host-side surface is therefore:

- `wp.LOG_DEBUG`, `wp.LOG_INFO`, `wp.LOG_WARNING`, `wp.LOG_ERROR` — level
  constants for comparing or assigning to `wp.config.log_level`.
- `wp.Logger` — runtime-checkable `Protocol` describing the four-method
  interface (`debug`, `info`, `warning`, `error`) that custom loggers
  implement. Frameworks supply any object satisfying the protocol; no
  inheritance required. This mirrors the `wp.Allocator` extension point.
- `wp.utils.LoggerKit` — Omniverse Kit / Carbonite integration (under `wp.utils`
  alongside `wp.utils.AllocatorRmm`, since it's a framework-specific
  implementation rather than a primary API).
- `wp.set_logger()`, `wp.get_logger()` — install/retrieve the active logger.
  Pass `None` to `set_logger()` to restore the built-in default. The default
  implementation is internal (`warp._src.logger.LoggerBasic`) and not part of
  the public API; users compose with the protocol or extend the default
  via `set_logger()`.
- `wp.ScopedLogger(logger)` — context manager that installs `logger` for the
  duration of the `with` block and restores the previous logger on exit
  (mirrors `wp.ScopedAllocator`).
- `wp.ScopedLogLevel(log_level)` — context manager that overrides
  `wp.config.log_level` for the duration of the `with` block and restores
  the previous value on exit.

There is no generic `wp.log(level, msg)` builtin on the kernel side either. The
level is encoded in the function name, which:

- Eliminates the need for a compile-time-constant validation (the level is
  implicit).
- Keeps the kernel-side API surface small and obvious.
- Avoids the false suggestion that the level could be a runtime variable.

#### Kernel-side signatures

Each of the four built-ins accepts a format string and optional variadic
payload arguments, using C-style printf format specifiers:

```text
wp.log_<level>(fmt: str)
wp.log_<level>(fmt: str, *args)
```

```python
# Examples
wp.log_info("iteration %d, residual %f", i, r)
wp.log_debug("position: %f %f %f", v[0], v[1], v[2])
wp.log_warning("unexpected value: %f", x)
```

The `fmt` argument must be a string literal (resolved at codegen time). The
format specifiers and argument types are validated at compile time; mismatches
are codegen errors. `%s` is a Warp extension that accepts vector payloads
(`vec2f`, `vec3f`, `vec4f`, `vec2d`).

**Why printf-style, not `{}`-style.** Warp already has `wp.printf()` in kernels
using C-style format specifiers. Using a different convention for `wp.log_*()`
would be inconsistent.

**Payload budget.** Each ring buffer entry reserves up to 128 bits (16 bytes) for
payload data across all arguments. Codegen parses the format string at compile
time to determine the number and types of arguments, and validates that the total
payload fits within the budget.

| Fits (up to 128 bits total)                        | Does not fit                    |
| -------------------------------------------------- | ------------------------------- |
| Up to 4x `int32`/`float32` (variadic composite)    | `mat33` (288 bits)              |
| Up to 2x `int64`/`float64` (variadic composite)    | `mat44` (512 bits)              |
| 1x `vec2f`, `vec3f`, `vec4f`, or `vec2d` (typed)   | More than 4 scalars             |

The variadic composite path accepts scalar arguments only (`int32`, `float32`,
`int64`, `float64`). Vector payloads use the typed single-payload overloads
(one vector value per call).

This makes each ring buffer entry ~24-32 bytes (with alignment) instead of the
12 bytes a single 32-bit payload would require. At the default capacity of 1024
entries the buffer is ~32 KB, which is trivial for GPU memory. Unsupported types
or payloads exceeding 128 bits are rejected at compile time (codegen error), not
silently truncated.

The 128-bit cap is a pragmatic cutoff: it covers the types users most commonly
want to inspect during debugging (scalars, positions, quaternions) without
unbounded entry sizes.

#### Host-side internal emitters

Warp's own diagnostics flow through internal helpers in `warp/_src/logger.py`:

```python
# warp/_src/logger.py — internal API, not exported
def log_debug(message: str) -> None: ...
def log_info(message: str) -> None: ...
def log_warning(message: str, category=None, stacklevel=1, once=False) -> None: ...
def log_error(message: str) -> None: ...
```

`log_warning` accepts extra parameters for Python warning filter integration.
These parameters do not exist on the kernel side (warnings filters are a
host-only concept).

Internal callers import these directly:

```python
from warp._src.logger import log_warning
log_warning("Array size mismatch")
```

### Level Filtering

#### `wp.config.log_level` (global default)

A single integer threshold on `warp.config`. Default is `LOG_INFO` (20).

- **Host-side:** The internal `log_*()` helpers check `config.log_level` before
  dispatching to the active logger. `log_error()` always emits.
- **Kernel-side:** `config.log_level` is synced to the root of the kernel logger
  hierarchy (`logging.getLogger("warp.kernel").setLevel(config.log_level)`).
  Records below the threshold are discarded at drain time.

#### Per-module override (kernel logs only)

Users who want fine-grained control over kernel log output can configure
individual loggers in the `warp.kernel.*` hierarchy using stdlib `logging`:

```python
import logging

wp.config.log_level = wp.LOG_WARNING              # quiet globally
logging.getLogger("warp.kernel.my_sim").setLevel(logging.DEBUG)  # verbose for one module
```

This works because `config.log_level` sets the *default* level on the parent
logger, not a hard floor. Per-logger configuration via stdlib `logging` takes
precedence, following standard Python `logging` semantics.

Host-side diagnostics do not support per-module override. `config.log_level` is
the sole gate for Warp's own messages, which is appropriate since users do not
author those messages.

#### Deprecated flags

`wp.config.verbose` and `wp.config.quiet` continue to work during the
deprecation window: setting either is honored alongside `log_level` at the
init banner and module-load timer call sites. `Runtime.__init__` emits a
one-time `DeprecationWarning` (via `log_warning(once=True)`) when either
flag is non-default, pointing users at `log_level` as the replacement.

`wp.config.verbose_warnings` is **not** deprecated; it is an orthogonal
formatting flag that controls whether warning output includes the source
location, and there is no equivalent `log_level` setting.

### Output Routing

#### Host-side

The pluggable `Logger` protocol routes output:

| Implementation       | debug/info          | warnings                            | errors              |
| -------------------- | ------------------- | ----------------------------------- | -------------------- |
| Default (internal)   | `sys.stdout`        | `warnings.warn()` -> `sys.stderr`   | `sys.stderr`         |
| `LoggerKit`          | `carb.log_verbose`/`carb.log_info` | `carb.log_warn()`      | `carb.log_error()`   |

The default logger routes warnings through `warnings.warn()` to respect
Python's filter machinery (`-W`, `simplefilter()`, per-module filters). Custom
formatting is applied via a temporary `showwarning` override within the logger.

`LoggerKit` routes all output through Omniverse Kit's Carbonite (`carb`)
logging functions instead of raw `print()`. This means Warp messages appear in
Kit's log viewer with proper severity levels, timestamps, and source
attribution. Kit has its own log filtering, so `LoggerKit` bypasses Python's
`warnings.warn()` machinery and sends warnings directly through `carb.log_warn()`.
The `carb` module is imported lazily since it is only available at runtime inside
a Kit process.

Both implementations format messages with the same `Warp <Level>: <msg>`
prefix (`Warp Error:`, `Warp UserWarning:`, etc.) so that Warp's output is
identifiable regardless of which logger is active.

Frameworks register a custom logger via `wp.set_logger()`.

#### Kernel-side

Kernel log records are written to a per-stream ring buffer in device memory
(with a pageable host staging buffer used for the D→H drain). Records are
drained to the host during `wp.synchronize()`, `wp.synchronize_device()`, or
`wp.synchronize_stream()`, then forwarded to
`logging.getLogger("warp.kernel.{module}")` using `makeRecord` so that source
location (filename, line number) resolves to the kernel source, not the drain
function.

Buffer capacity is configured via `wp.config.kernel_log_capacity` (default:
1024 entries). Overflow is tracked atomically and reported as a warning after
drain.

The host and kernel sides intentionally use different abstractions: the host
side uses the `wp.Logger` Protocol and routes warnings through Python's
`warnings` machinery (so filters like `warnings.filterwarnings()` and `-W`
flags work), while kernel logs go through stdlib `logging` to support
per-module configuration of GPU-emitted records. Migrating the host side to
stdlib `logging` is not on the roadmap.

## Testing Strategy

- **Host-side (`test_logger.py`):** Verify level gating, warning filter
  integration, default-logger and `LoggerKit` output routing, `set_logger()` /
  `get_logger()`. Tests import the default `LoggerBasic` and the internal
  emitters directly from `warp._src.logger`.
- **Kernel-side:** Verify each of the four named builtins with and without
  payloads, overflow detection, drain on sync, per-module logger routing.
  Device coverage: CPU and CUDA.
- **Integration:** Verify that `config.log_level` gates both host and kernel
  output. Verify that per-module logger overrides work independently of the
  global threshold.
