import os
import sys
from pathlib import Path

from pytype.tools.analyze_project.main import main as pytype
from retype import retype_file as retype
from retype import ReApplyFlags

from function_inliner import inline_funcs
from gen_hdfg import extract_dfg
from propagator import propagate
from add_shapes import add_shapes

import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be distilled')
    parser.add_argument(
        '-d',
        '--disable-adding-shapes',
        action='store_true',
        dest='disable_adding_shapes',
        help='disable adding shape info to stub files'
    )
    parser.add_argument(
        '-t',
        '--disable-dfg-transformation',
        action='store_true',
        dest='disable_dfg_transformation',
        help='disable dfg transformation'
    )
    args = parser.parse_args()

    base = os.path.basename(args.pythonfile)

    print("---------inline function and rename variables---------")
    inline_funcs(args.pythonfile)

    pytype_input = 'distill_output/dfg_transformation/'+base

    if args.disable_dfg_transformation:
        pytype_input = 'distill_output/function_inline/'+base
    else:
        print("---------perform DFG transformation---------")
        dfg_transform_input = 'distill_output/function_inline/'+base
        extract_dfg(dfg_transform_input)

    print("---------generate .pyi file-------------")
    pytype([pytype_input])

    if args.disable_adding_shapes:
        retype_pyi_input = 'distill_output/pytype/pyi'
    else:
        # add shape information to the .pyi files
        # dynamic typing will re-run argparse if it's part of a script, which will mess up our args
        print("----------add shapes-----------")
        args_copy = argparse.Namespace(**vars(args))
        add_shapes(pytype_input, globals(), locals())
        args = args_copy

        retype_pyi_input = 'distill_output/monkeytype/pyi'

    print("----------merge .pyi file with src----------")
    retype_src_input = pytype_input
    retype_target_input = 'distill_output/retype'
    retype(
        Path(retype_src_input),
        Path(retype_pyi_input),
        Path(retype_target_input),
        flags=ReApplyFlags(replace_any=False, incremental=True)
    )

    print("----------propagate types from callee to caller--------")
    propagate_input = 'distill_output/retype/' + base
    propagate(propagate_input)
