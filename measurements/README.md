# Measurement records

One JSON file per measurement, in the format of `schema.json`. Each record carries the simulator's own
machine and engine JSON, how the number was taken, and the medians, so it is reproducible by construction.

**Make one in the app:** on the engine card, fill the *Measured on real hardware* block, click
*Export measurement*, answer the short method form, download. The file is validated before download: parts
must exist in the catalogs, method fields must be present, and the numbers must stay inside physical bounds
(a card cannot write faster than its memory bus allows).

**From a result file:** `python3 bench/to-measured.py results.jsonl` gives the results block; the rest is the
config you ran with.

**Submit:** a pull request that adds the file to this folder (file name = the id the app gives it). Keep
`method.submitter` an alias and free text without names, addresses or links; the validator warns on those.

**What happens with it:** the build copies all records into the site. The browser build aggregates them into
clusters keyed by GPUs, engine, model, quant and streams; on any engine card whose configuration matches a
cluster, the app offers the cluster's median as a one-click measured value, with the count and spread. The
Calibration page lists the clusters with the model's error per cluster. Records never change the model's
constants by themselves; promoting a cluster into the calibration set is a deliberate maintainer step.

Trust score (shown with each cluster): share of method fields present, +0.15 for three or more runs, +0.10 for
a raw-file hash.
