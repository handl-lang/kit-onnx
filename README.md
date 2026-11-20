# kit-onnx

The ONNX backend for the mlopt game (ADR 0006).

`kit_onnx` implements the `mlopt_core.ports` `Sink`/`Source` ports over
`onnx` + `onnxruntime`: `OnnxSink` lowers mlopt IR to a runnable ONNX
artifact, `OnnxExecutor` runs it, `OnnxSource` lifts an ONNX graph back
to IR — every path verified through onnxruntime.

It is a **leaf**: it depends on `mlopt-core` + `mlopt-dtypes`
(handl-lang/eqsat) and the external `onnx`/`onnxruntime`/`numpy`
stack — no torch, no `kit_torch`, no `kit_carriers`, no
`mlopt_orchestrator`.

## Develop

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
uv run scripts/gate.py --coverage   # pytest → ruff → ty → lint-imports → coverage floor
```

`scripts/gate.py` is a stdlib runner over the dev toolchain; `ci.sh`
is the gate entry.

## License

MIT.
