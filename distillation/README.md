# Distillation

This directory contains code of distillation.

# Usage
```
cd pytype && pip install .
cd ../retype && pip install .
cd ../MonkeyType && pip install .
cd ..

# DFG Transformation & Type Inference
python distll.py examples/signal_proc_mcube_np.py

# Check Purity and add pure decorator (In progress)
python check_pure.py -w examples/signal_proc_mcube_np.py

```

# Outputs (in `distill_outputs/`)
- `dfg_transformation/`
  - `signal_proc_mcube_np.py`: transformed code
  - `signal_proc_mcube_np.py.png`: visualized DFG
- `pytype/pyi/signal_proc_mcube_np.pyi`: stub files
- `monkeytype/pyi/signal_proc_mcube_np.pyi`: stub files with shapes
- `retype/signal_proc_mcube_np.py`: temp typed python code (only globals typed)
- `propogate/signal_proc_mcube_np.py`: fully typed python code (Both globals & function signatures typed)
