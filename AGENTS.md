# Agent guide — kit-onnx

The ONNX backend for the mlopt game (ADR 0006), split from
`handl-lang/graphopt` `packages/mlopt-onnx` in P3 Step D-1 (the first
package of the graphopt domain-package dissolution; the
`mlopt-*` → `kit-*` rename).  Python 3.13, stdlib + `onnx` +
`onnxruntime` + `numpy` on top of the eqsat substrate.

## Layout

- `src/kit_onnx/` — the one shipped package:
  - `sink.py` — `OnnxSink` lowers mlopt IR to ONNX (`lower_ir`,
    `GraphBuilder`);
  - `opmap.py` — the IR → ONNX op table (`ONNX_BINDINGS`);
  - `arrays.py` — dtype/array conversions (`DTYPE_NAMES`, `to_numpy`,
    `verify_module` helpers);
  - `executor.py` — `OnnxExecutor`, the `onnxruntime` runner;
  - `lift.py` / `liftmap.py` — `OnnxSource` + `lift_model`: ONNX → IR;
  - `lower.py` — IR → ONNX graph builder;
  - `report.py` — verify/equivalence reporting;
  - `errors.py` — `OnnxImportError` / `OnnxLoweringError` /
    `UnsupportedOnnxOpError`;
  - `register()` — seats `"onnx"` on `mlopt_core.backends`'s
    `BackendRegistry` (explicit; importing registers nothing).
- `tests/` — the package-bound suite (`test_onnx_opmap.py`,
  `test_onnx_backend_registry.py`).  The cross-backend integration
  tests (`test_onnx_sink.py`, `test_onnx_source.py`,
  `test_onnx_liftmap.py`, `test_onnx_codec_drift.py`,
  `test_dtype_registry.py`) stayed in graphopt — they exercise
  `kit_onnx` *through* `mlopt_torch`/`mlopt_orchestrator`/
  `mlopt_host_tensor` and pin this repo as a git dep.

## Contracts

- **Leaf by contract** — no torch, no `mlopt_torch`, no
  `mlopt_carriers`, no `mlopt_orchestrator` imports; the
  `onnxruntime` boundary is the only external engine.
- Substrate deps: `mlopt-core`, `mlopt-dtypes` (handl-lang/eqsat,
  `packages/` subdirs, tag-pinned).
- History was preserved via `git subtree split` of
  `graphopt:packages/mlopt-onnx` — `git log` here reaches back
  through the graphopt history.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

- Coverage floor is `fail_under = 43` in `[tool.coverage.report]`
  (measured 43.12% on the package-bound suite — `lift.py`/`liftmap.py`/
  `opmap.py` get their heavy exercise through graphopt's suite via the
  git dep).  Tighten, never lower.
- The leaf-by-contract layering is enforced statically by the
  import-linter `forbidden` contract: `kit_onnx` must not import
  torch / `mlopt_torch` / `mlopt_carriers` / `mlopt_orchestrator` /
  `mlopt_host_tensor` / `mlopt_cuda` (torch appears only in `tests/`,
  outside `root_packages`).

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
