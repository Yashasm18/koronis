# Koronis — hosted demo

`index.html` here is generated, not hand-edited. Rebuild it with:

```bash
python site/build.py
```

It embeds `results/replay_demo.json`, `seeds_summary.csv`, `mechanism.csv`,
`frontier.csv` and `benchmark.json` verbatim, so the page always shows whatever
the last experiment run actually produced. The source lives in `site/`.
