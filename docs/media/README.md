# README media

`rocking_q.png` is drawn by `make_figure.py` from this commit's code, with no measured data:

```sh
uv run --with matplotlib docs/media/make_figure.py
```

It takes the seeded volume of `examples/synthetic_scan.py` (`make_volume()`, seed 0) and the same
Theta steps, and bins it with `map_rocking_scan` at delta 60°, gamma 0°, 9 keV, 55 µm pixels and
1000 mm, the values `tests/test_rocking.py` uses. The panels are the middle frame, the summed ROI
counts per frame, and the q-space map averaged over the measured q_y bins. Running it again gives
the same file byte for byte (checked on Windows, matplotlib 3.11.2).
