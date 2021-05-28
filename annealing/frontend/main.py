import sys
import argparse
import typed_ast.ast3 as ast
import typed_astunparse as astunparse
import util.error as err

def parse_args():
    ap = argparse.ArgumentParser(description = 'Intrepydd compiler')
    ap.add_argument('file', help = 'input file (must be *.pydd)')
    ap.add_argument('-o', default = '', help = 'Output file name (must be *.py)')
    ap.add_argument('-verbose-level', default = 3, help = 'Verbose level (default 3)')
    ap.add_argument('-const-annot', default = False, action = 'store_true', help = 'Presume constant sizes in annotations')
    ap.add_argument('-topfunc-fusion', default = False, action = 'store_true', help = 'Fuse top-level functions for co-optimization')
    ap.add_argument('-poly-opt', default = False, action = 'store_true', help = 'Enable polyhedral optimizations')
    ap.add_argument('-poly-opt-layout', default = False, action = 'store_true', help = 'Enable data layout optimizations in poly-opt')
    ap.add_argument('-ray-pfor', default = False, action = 'store_true', help = 'Ray task parallelism for parallel loop (pfor)')
    ap.add_argument('-ray-pfor-schedule', default = 'block', help = 'Ray pfor\'s scheduling policy (block or cyclic, default block)')
    ap.add_argument('-ray-pfor-cupy', default = False, action = 'store_true', help = 'Ray pfor to use cupy (whenever possible)')
    ap.add_argument('-O0', default = False, action = 'store_true', help = 'Optimization level 0')
    ap.add_argument('-O1', default = False, action = 'store_true', help = 'Optimization level 1')
    ap.add_argument('-O2', default = False, action = 'store_true', help = 'Optimization level 2')
    ap.add_argument('-O3', default = False, action = 'store_true', help = 'Optimization level 3')
    args = ap.parse_args()
    return args

def check_and_get_filenames(name_in, name_out):
    if len(name_in) >= 4 and name_in[-3:] == '.py':
        python_input = True
    elif len(name_in) >= 6 or name_in[-5:] == '.pydd':
        python_input = False
    else:
        err.error('Intput file name must be *.pydd or *.py')

    if not name_out:
        if python_input:
            name_out = name_in.replace('.py', '_pydd_kernel.py')
            name_pyout = name_in.replace('.py', '_pydd_main.py')
        else:
            name_out = name_in.replace('.pydd', '_opt.py')
            name_pyout = ''
    else:
        if python_input:
            name_pyout = name_in.replace('.py', '_pydd_main.py')
        else:
            name_pyout = ''

    if len(name_out) < 4 or name_out[-3:] != '.py':
        err.error('Output file name must be *.py, but found:', name_out)

    return name_in, name_out, name_pyout, python_input

def parse():
    args = parse_args()
    name_in, name_out, name_pyout, python_input = check_and_get_filenames(args.file, args.o)

    file_in = open(name_in, 'r')
    source = file_in.read()
    file_in.close()
    tree = ast.parse(source)

    opts = { 'name_in': name_in, 'name_out': name_out,
             'name_pyout': name_pyout, 'python_input': python_input, }
    opts['verbose_level'] = int(args.verbose_level)
    opts['const_annot'] = args.const_annot
    opts['topfunc_fusion'] = args.topfunc_fusion
    opts['poly_opt'] = args.poly_opt
    opts['poly_opt_layout'] = args.poly_opt_layout
    opts['ray_pfor'] = args.ray_pfor
    opts['ray_pfor_schedule'] = args.ray_pfor_schedule
    opts['ray_pfor_cupy'] = args.ray_pfor_cupy

    olv = 3 if args.O3 else 2 if args.O2 else 1 if args.O1 else 0
    if olv >= 1:
        opts['topfunc_fusion'] = True
        opts['poly_opt'] = True
        opts['poly_opt_layout'] = True
    if olv >= 2:
        opts['ray_pfor'] = True
        opts['ray_pfor_schedule'] = 'cyclic'
    if olv >= 3:
        opts['ray_pfor_cupy'] = True

    return tree, opts


if __name__ == "__main__":
    tree, opts = parse()
    print('tree:', ast.dump(tree))
    print('opts:', opts)
    print('unparsed source:', astunparse.unparse(tree))
