import typed_ast.ast3 as ast
import util.error as err

def extract_value(exp):
    if isinstance(exp, ast.Num):
        return exp.n
    if isinstance(exp, ast.Str):
        return exp.s
    if isinstance(exp, ast.NameConstant):
        return exp.value
    elif isinstance(exp, ast.Name):
        return exp.id
    else:
        return exp

def name(var, is_store = False):
    return ast.Name(id = str(var), ctx = ast.Store() if is_store else ast.Load())

def assign(var, expr, comment = None):
    if isinstance(var, tuple) or isinstance(var, list):
        targets = [name(v, True) for v in var]
    else:
        targets = [name(var, True)]
    return ast.Assign(targets = targets, value = expr, type_comment = comment)

def init_array_statement(array, atype, elm_module = 'numpy', init_func = 'zeros'):
    import libbase.entry as libkb
    pkg1 = libkb.module_to_var(atype.module)
    if not pkg1:
        err.error('Cannot find (alias) name for imported module:', atype.module)
    pkg2 = libkb.module_to_var(elm_module)
    if not pkg2:
        err.error('Cannot find (alias) name for imported module:', elm_module)
    att1 = ast.Attribute(value = ast.Name(id = pkg1, ctx = ast.Load()), attr = init_func)
    att2 = ast.Attribute(value = ast.Name(id = pkg2, ctx = ast.Load()), attr = str(atype.etype))

    sizes = [ast.Name(id = str(s), ctx =ast.Load()) for s in atype.shape]
    tpl = ast.Tuple(elts = sizes, ctx = ast.Load())
    v = ast.Call(func = att1, args = [tpl, att2], keywords = [])
    ret = ast.Assign(targets = [ast.Name(id = array, ctx = ast.Store())],
                     value = v, type_comment = None)
    return ret
