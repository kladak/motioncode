# Data provenance

## Synthetic (default, and what CI runs)

`motioncode/data/synthetic.py` generates every waveform on a fixed seed, with four
morphology labels: `regular`, `irregular`, `wide` and `burst`. The numbers in the README
and in `reports/metrics.json` come from that generator.

## PhysioNet (optional, not used by CI)

`scripts/download_physionet.py` fetches MIT-BIH for experimentation after
`pip install '.[physionet]'`. Mapping MIT-BIH annotation symbols onto the four simulator
morphology labels is left for a later slice, since the two labelling schemes describe
different things. Metrics in this repository come from the generator.
