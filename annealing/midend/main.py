import typed_ast.ast3 as ast
import midend.typeinfer as typeinfer
import midend.polyhedral as polyhedral
import midend.arrayopt as arrayopt
import util.error as err

def optimize(tree, opts):
    err.set_verbose_level(opts['verbose_level'])

    typeinfer.infer(tree, opts)
    # Todo: split & propagate statements (for polyopt and arropt)
    if opts['poly_opt']:
        polyhedral.optimize(tree, opts)
    arrayopt.optimize(tree, opts)

if __name__ == "__main__":
    import frontend.main as fe
    tree, opts = fe.parse()
    optimize(tree, opts)
    # print('tree:', ast.dump(tree))
    # print('opts:', opts)
