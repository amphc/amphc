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

## Publications
* "Intrepydd: Performance, productivity, and portability for data science application kernels".
  Tong Zhou, Jun Shirako, Anirudh Jain, Sriseshan Srikanth, Thomas M. Conte, Richard Vuduc, and Vivek Sarkar. 
  In Proceedings of the 2020 ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software (Onward! 2020).
  pages 65–83. 2020.

