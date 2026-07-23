# Executed Notebook Snapshots

These notebooks contain the inline figures and tables used for reading the
pre-class notes without running training first. They are generated from the
clean source notebooks one directory above by `scripts/run_toy_notes.py` and
should not be edited directly.

The first six snapshots are executed together with fixed seeds. Notebook
[06, the CIFAR-10 image experiment](cifar10_steering_with_unconditional_edm.ipynb),
is executed separately because it requires a pretrained EDM checkpoint, fitted
PCA statistics, a frozen evaluator, and a CUDA GPU. The source notebooks remain
the canonical code; these copies are the viewable publication artifacts.

The published snapshots were executed on an NVIDIA RTX A6000 and include their
inline figures and tables.
