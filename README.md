# The Framework for Deep Learning

## TODO
### [Phase 1 - FRAMEWORK CORE](https://github.com/Khoawawa/DeepLearning_Framework/tree/phase_1)
#### Critical
- [x] Currently the trainer and callback interface are blocking each other via cyclical referencing (see `icommon.py` and `itrainer.py`)
- [x] Move data to the correct device after loading in `trainer.fit()` 
- [ ] Build simple trainer for unit testing --> [KhoaNA]
- [ ] Build simple dataset for unit testing --> [ThinhNHH]
#### Medium
- [x] Refactor build trainer and factory
- [ ] `DataLoader` batch type: `training_step(self, x: torch.Tensor, ...)` type hint may not hold if batches become dicts/tuples later.
#### Low
- [ ] Build ITester --> [HyNX]
- [ ] Build simple Tester implementing ITester for unit testing --> [HyNX]
- [ ] Mid-epoch resuming: currently only support epoch resume, consider step-triggered saves from checkpoint callback `on_step_end`
- [ ] Confirm exit-code handling won't break if `main.py` is ever invoked from a Colab/notebook context (per portability goal) — `SystemExit` behaves differently there.
