# Annealing

## Set up
Dependence:
* Python3
  * typed_ast
  * typed_astunparse
  * islpy
  * sympy
  * numpy
  * ray (to run the generated code using Ray runtime)
  * cupy (to run the generated code on GPUs)

Need to add the top directory to PYTHONPATH, e.g., `export PYTHONPATH=$PYTHONPATH:$PWD`

## Compile
```
$ pyddc -O3 program.py -o program_opt.py
```

## Run
```
$ python program_opt.py
```

## Note
* Link to Intrepydd HP: https://hpcgarage.github.io/intrepyddguide/
