import typed_ast.ast3 as ast
import util.error as err

new_var_count = 0
new_var_base = '__var'
def new_var(vars):
    found = True
    while found:
        global new_var_count, new_var_base
        new_var_count += 1
        name = new_var_base + str(new_var_count)
        found = name in vars
    if isinstance(vars, set):
        vars.add(name)
    else:
        err.error('Unsupported variable group:', type(var), vars)
    return name
