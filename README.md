# kit-onnx

The ONNX backend for the mlopt game (ADR 0006), split out of
`handl-lang/graphopt` (`packages/mlopt-onnx`) in P3 Step D-1 and
renamed `mlopt-onnx` → `kit-onnx` / `mlopt_onnx` → `kit_onnx`.

`kit_onnx` implements the `mlopt_core.ports` `Sink`/`Source` ports over
`onnx` + `onnxruntime`: `OnnxSink` lowers mlopt IR to a runnable ONNX
artifact, `OnnxExecutor` runs it, `OnnxSource` lifts an ONNX graph back
to IR — every path verified through onnxruntime.

It is a **leaf**: it depends on `mlopt-core` + `mlopt-dtypes`
(handl-lang/eqsat) and the external `onnx`/`onnxruntime`/`numpy`
stack — no torch, no `mlopt_torch`, no `mlopt_carriers`, no
`mlopt_orchestrator`.

## Install

```sh
uv sync --group dev
```

or as a git dependency:

```toml
kit-onnx = { git = "https://github.com/handl-lang/kit-onnx.git", tag = "v0.1.0" }
```

Substrate pins (`mlopt-core`, `mlopt-dtypes`) resolve from
`handl-lang/eqsat` — see that repo's tags.

## Verify

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src tests
.venv/bin/ty check
```
