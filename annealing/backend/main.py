import typed_ast.ast3 as ast
import typed_astunparse as astunparse
import backend.raypfor as raypfor
import backend.pythonizer as pythonizer
import util.error as err

def codegen(tree, opts):
    if opts['ray_pfor']:
        raypfor.transform(tree, opts)

    codegen_sequence(tree, opts, opts['name_out'])

    if opts['python_input']:
        assert isinstance(tree, ast.Module)
        code = 'from %s import *' % opts['name_out'].replace('.py', '')
        tree2 = ast.parse(code)
        tree2.body += tree.body_py
        codegen_sequence(tree2, opts, opts['name_pyout'])

def codegen_sequence(tree, opts, name_out):
    pythonizer.transform(tree, opts)
    output = astunparse.unparse(tree)
    file_out = open(name_out, 'w')
    file_out.write(output)
    file_out.close()
