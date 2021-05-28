import typed_ast.ast3 as ast
from typed_astunparse import unparse
import util.error as err
import libbase.entry as libkb

def optimize(tree, opts):
    ArrayOptimization().visit(tree)

class ArrayOptimization(ast.NodeVisitor):
    def __init__(self):
        pass
