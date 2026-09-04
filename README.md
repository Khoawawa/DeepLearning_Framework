# The Framework for Deep Learning

## TODO
### [Phase 1 - FRAMEWORK CORE](https://github.com/Khoawawa/DeepLearning_Framework/tree/phase_1)
#### Blocking
- [ ] Create missing `configs/data/imagenet_top1k.yaml` (or fix the reference in `config.yaml` if the filename is wrong) — currently crashing `main.py` at config load time.
#### BaseTrainer plumbing (`engines/trainer.py`)
- [ ] Decide whether `fit()` should move `batch` to `self.device` before calling `training_step`, or leave that responsibility to the caller/subclass.
- [ ] `load_state_dict` on subclasses will throw a raw `KeyError` if a checkpoint doesn't match the current component set (e.g. saved from a different model variant). Consider `strict=False` + logged mismatch warning once multiple model variants exist.
#### Checkpointing
- [ ] Mid-epoch resume: currently only epoch-granularity resume is reliable — step-triggered saves record a checkpoint mid-epoch but `current_epoch` isn't incremented until the epoch fully completes, so resuming replays the whole epoch while `global_step` keeps climbing.
  - Option A (cheap): treat step-triggered saves as crash-insurance only, not true resume points.
  - Option B (proper): track `batch_in_epoch`, use a resumable sampler or skip-N logic in `fit()`.
- [ ] Make `CheckpointCallBack._save` atomic (write to temp file + `os.replace`) so a crash mid-save can't leave a corrupted `latest.pt`.
#### Typing / protocols (`icommon.py`)
- [x] `TorchModule.to/train/eval` now return `Self` instead of `TorchModule` (needs `typing_extensions.Self` if not on Python 3.11+).
#### Config resolution (`common_utils.py`)
- [x] `resolve_config_path` now normalizes `.yaml`/`.yml` extensions instead of blindly appending `.yaml`.
- [ ] Decide: should a wrong extension (e.g. `config.txt`) raise, or silently coerce to `.yaml`? Leaning toward raising given the future web-app-generated-config use case.
- [ ] Remove the internal `logger.error(...)` inside `resolve_config_path` (and similar low-level functions) now that `main()`'s top-level handler already logs errors with file/line — currently duplicated in terminal output.
#### main.py / logging
- [x] Terminal vs. file log split via loguru (INFO+ compact to terminal, DEBUG+ with full traceback to file).
- [x] Switched from `sys.exit(1)` inside `try/except` to `return 1` + `raise SystemExit(main())` at module level, avoiding the double-traceback.
- [ ] Confirm exit-code handling won't break if `main.py` is ever invoked from a Colab/notebook context (per portability goal) — `SystemExit` behaves differently there.
#### Other / deferred
- [ ] `DataLoader` batch type: `training_step(self, x: torch.Tensor, ...)` type hint may not hold if batches become dicts/tuples later.
---
