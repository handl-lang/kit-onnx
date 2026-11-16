"""ONNX backend for mlopt — ADR 0006, both directions.

The first non-torch backend, now complete.  ``OnnxSink`` implements
``mlopt_core.ports.Sink`` (direction 1: IR -> ONNX, verified through
``onnxruntime``); ``OnnxSource`` implements
``mlopt_core.ports.Source`` (direction 2: ONNX graph -> IR, over
:mod:`kit_onnx.liftmap` — the inverse op vocabulary of the sink's
:mod:`kit_onnx.opmap`).  Together they close the loop the ADR
staged: ONNX models enter the optimizer untouched by torch, and
optimized programs leave for any ONNX runtime.

The experiment this package *is*: ADR 0003/0006 claim that a new
backend implements the ports and nothing in ``mlopt-core`` changes.
The port boundary is what is under test, so the honest reading is
that the code here is half the deliverable and the answer to "did
core have to move?" is the other half.

    >>> from kit_onnx import OnnxSink, OnnxSource
    >>> source = OnnxSource()                # doctest: +SKIP
    >>> ir, leaves = source.to_ir(model)     # doctest: +SKIP
    >>> sink = OnnxSink(dtype="float64",
    ...                 input_dtypes=source.input_dtypes)  # +SKIP
    >>> executor = sink.lower(ir, leaves)    # doctest: +SKIP

Nothing here imports torch, ``kit_torch``, ``kit_carriers`` or
``mlopt_orchestrator``: the ONNX backend is a *peer* of the torch
one, reachable through the same ports and depending only on
``mlopt_core``.
"""

from __future__ import annotations

from kit_onnx.arrays import (
    ONNX_TYPE_NAMES,
    WORKING_DTYPES,
    numpy_to_onnx_type,
    to_numpy,
)
from kit_onnx.errors import (
    OnnxImportError,
    OnnxLoweringError,
    UnsupportedOnnxOpError,
)
from kit_onnx.executor import OnnxExecutor
from kit_onnx.lift import GraphLifter, LiftResult, lift_model
from kit_onnx.liftmap import ONNX_LIFTS
from kit_onnx.lower import GraphBuilder, lower_ir
from kit_onnx.opmap import ONNX_BINDINGS
from kit_onnx.report import (
    REL_FLOOR,
    OnnxVerifyReport,
    verify_equiv,
    verify_module,
)
from kit_onnx.sink import OnnxSink
from kit_onnx.source import MAX_OPSET, MIN_OPSET, OnnxSource

__all__ = [
    "MAX_OPSET",
    "MIN_OPSET",
    "ONNX_BINDINGS",
    "ONNX_LIFTS",
    "ONNX_TYPE_NAMES",
    "REL_FLOOR",
    "WORKING_DTYPES",
    "GraphBuilder",
    "GraphLifter",
    "LiftResult",
    "OnnxExecutor",
    "OnnxImportError",
    "OnnxLoweringError",
    "OnnxSink",
    "OnnxSource",
    "OnnxVerifyReport",
    "UnsupportedOnnxOpError",
    "lift_model",
    "lower_ir",
    "numpy_to_onnx_type",
    "register",
    "to_numpy",
    "verify_equiv",
    "verify_module",
]


def register() -> None:
    """Seat the ONNX backend on ``mlopt_core.backends.DEFAULT_REGISTRY``.

    Explicit seam — importing this package registers nothing; call
    ``kit_onnx.register()`` at the entry point (the same pattern as
    ``mlopt_kernel.play.register()``).  Registers ``"onnx"`` with
    :class:`OnnxSource` as the source and ``OnnxSink`` as the ``"onnx"``
    target — both lazy factories, resolved on demand.  ONNX has no
    composer/meter yet, so ``build_backend`` reports them missing
    honestly; ``find_backends_by_ops`` capability queries work today.
    """
    from mlopt_core import backends

    backends.register_backend(
        "onnx",
        source=OnnxSource,
        targets={"onnx": OnnxSink},
        notes="ONNX source+sink (ADR 0006); composer/meter pending",
    )
