import typed_ast.ast3 as ast
import typed_astunparse as astunparse
import util.error as err

def transform(tree, opts):
    TypeAnnotRemover().visit(tree)

class TypeAnnotRemover(ast.NodeVisitor):
    def __init__(self):
        self.using_any = False

    def visit_Module(self, N):
        self.generic_visit(N)
        if self.using_any:
            im_any = ast.parse('from typing import Any').body[0]
            assert isinstance(im_any, ast.ImportFrom)
            N.body.insert(0, im_any)

    def visit_FunctionDef(self, N):
        N.returns = None
        self.generic_visit(N)

    def visit_arg(self, N):
        N.annotation = None

    def visit_AnnAssign(self, N):
        N.annotation = ast.Name(id = 'Any', ctx = ast.Load())
        self.using_any = True
