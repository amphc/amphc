import ast
import astunparse
import os
from pathlib import Path

import numpy as np

from purity_kb import builtin_pure_funcs, numpy_pure_funcs

kb = builtin_pure_funcs + numpy_pure_funcs
src_func_purity_dict = {}


class PureVisitor(ast.NodeVisitor):
    def __init__(self, visited):
        super().__init__()
        self.pure = True
        self.visited = visited

    def visit_Name(self, node):
        return node.id

    def visit_Attribute(self, node):
        name = [node.attr]
        child = node.value
        while child is not None:
            if isinstance(child, ast.Attribute):
                name.append(child.attr)
                child = child.value
            else:
                try:
                    name.append(child.id)
                except AttributeError as e:
                    # print('AttributeError:', e)
                    pass # 'Call' object has no attribute 'id'
                break
        name = ".".join(reversed(name))
        return name

    def visit_Call(self, node):
        if not self.pure:
            return
        name = self.visit(node.func)
        if name not in self.visited:
            self.visited.append(name)
            if name in src_func_purity_dict:
                if not src_func_purity_dict[str(name)]:
                    print(name, "is not pure from src_func_purity_dict")
                    self.pure = False
            else:
                try:
                    callee = eval(name)
                    if not is_pure(callee, self.visited):
                        print(name, "is not pure because of undefined purity")
                        self.pure = False
                except (NameError, TypeError) as e:
                    print(name, "is not pure because of NameError or TypeError:", e)
                    # TODO: TypeError: eval() arg 1 must be a string, bytes or code object
                    # Why name is None if there is no pure attribute? (for __add__ method?)
                    self.pure = False


class add_pure(ast.NodeTransformer):
    def __init__(self, pure_dict):
        super().__init__()
        self.pure_dict = pure_dict

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body.insert(0,
                         ast.Import(names=[ast.alias(name="distill_util", asname=None)])
                         )
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.pure_dict.get(node.name, False):
            node.decorator_list.append(
                ast.Name(id='distill_util.pure', ctx=ast.Load())
            )
        return node


def check_purity_in_kb(f):
    if f in kb:
        return True
    else:
        # print('Cannot find', f, 'in purity knowledge base.')
        raise NameError('Cannot find '+ str(f) + ' in purity knowledge base.')

import inspect, textwrap

def is_pure(f, _visited=None):
    # check function attribute (or pure decorator)
    try:
        return f.pure
    except AttributeError:
        pass

    # check knowledge base
    try:
        return check_purity_in_kb(f)
    except NameError as e:
        # print('Check in kb:', e)
        pass

    try:
        code = inspect.getsource(f.__code__)
    except AttributeError:
        return False
    except OSError:
        return False # could not get source code

    code = textwrap.dedent(code)
    node = compile(code, "<unknown>", "exec", ast.PyCF_ONLY_AST)

    if _visited is None:
        _visited = []

    visitor = PureVisitor(_visited)
    visitor.visit(node)
    return visitor.pure

def linenumber_of_function(f):
    try:
        return f[1].__code__.co_firstlineno
    except AttributeError:
        return -1


def check_pure(pythonfile, write_flag):
    MODULE_PATH = pythonfile # "./examples/input_file.py"
    MODULE_NAME = "mymodule"
    from importlib import util
    spec = util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
    module = util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    functions = inspect.getmembers(module, inspect.isfunction)
    functions.sort(key=linenumber_of_function)

    for func in functions:
        name = func[0]
        f = func[1]
        purity = is_pure(f)
        src_func_purity_dict[name] = purity
        print(name, purity)

    if write_flag:
        #transform and ouput a new file
        ast_node_purified = None
        with open(pythonfile) as pf:
            ast_node = ast.parse(pf.read())
            print(ast.dump(ast_node))
            ast_node_purified = add_pure(src_func_purity_dict).visit(ast_node)

        target_dir = Path('distill_output/purity_annotation/')
        target_dir.mkdir(parents=True, exist_ok=True)
        src = Path(pythonfile)
        with open(target_dir / src.name, "w") as out_file:
            out_file.write(astunparse.unparse(ast_node_purified))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed (full/relative path)')
    parser.add_argument('-w', action='store_true', help='a *_purified.py file will be written')

    args = parser.parse_args()
    check_pure(args.pythonfile, args.w)


