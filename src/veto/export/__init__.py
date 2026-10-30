"""Export a trained Veto model to a deployable artifact (Core ML).

The exported artifact is the boundary between this repository and any app that consumes the
model: ``dist/coreml/{veto.mlpackage, metadata.json, tokenizer/}``. Importing this package
does not require ``coremltools``; only the conversion and verification steps do.
"""
