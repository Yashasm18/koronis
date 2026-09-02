# Koronis — documentation

| | |
|---|---|
| [Architecture](architecture.md) | how it works, and where a learned model is and is not used |
| [AI decisions](ai-decisions.md) | every model choice, the ones rejected, and the measurement behind each |
| [Evaluation](evaluation.md) | protocol, ablations, calibration, forecasting, drift, latency |
| [Limitations](limitations.md) | what is assumed rather than measured, and what production would need |
| [Engineering log](engineering-log.md) | repository map, and every defect that changed a result |

[← back to the README](../README.md)

## The hosted demo

`index.html` in this directory is **generated, not hand-edited**:

```bash
python site/build.py       # results/ -> docs/index.html
```

It embeds the result artifacts verbatim, so the page always shows whatever the last
experiment run actually produced. The source lives in [`site/`](../site).
