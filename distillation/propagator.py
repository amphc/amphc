import sys
import re
import tokenize
from pathlib import Path

from lib2to3 import pygram, pytree
from lib2to3.pgen2 import driver, token
from lib2to3.pgen2.parse import ParseError
from lib2to3.pygram import python_symbols as syms
from lib2to3.pytree import Leaf, Node


def propagate(src):
    """Given an input python file `src`, propagate types from caller to caller and save in `target_dir`"""
    src = Path(src)
    target_dir = Path('distill_output/propagate')

    src = src.absolute()
    with tokenize.open(src) as src_buffer:
        src_contents = src_buffer.read()
        if src_contents == "":
            return
        src_encoding = src_buffer.encoding
        src_node = lib2to3_parse(src_contents)
        propagate_types(src_node)
    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / src.name, "w", encoding=src_encoding) as target_file:
        target_file.write(lib2to3_unparse(src_node))


def propagate_types(lib2to3_node):
    # collect annotation for all variables in the main call trace
    var_anno_dict = {} # {variable: corresponding annotation}
    func_caller_dict = {} # {function name: caller expr}
    for child in flatten_some(lib2to3_node.children):
        if child.type != syms.simple_stmt:
            continue
        # expr_stmt: testlist_star_expr (annassign | augassign (yield_expr|testlist) |
        #              ('=' (yield_expr|testlist_star_expr))*)
        # annassign: ':' test ['=' test]
        maybe_expr = child.children[0]
        if maybe_expr.type != syms.expr_stmt:
            continue

        expr = maybe_expr.children
        # Leaf(token.NAME, name.id)
        if (
                expr[0].type in (token.NAME, syms.power)
                and expr[1].type == syms.annassign
        ):
            if len(expr[1].children) > 2 and expr[1].children[2] != _eq:
                raise NotImplementedError(
                    f"unexpected element after annotation: {str(expr[3])}"
                )
            # save annotation in var_anno_dict
            var_anno_dict[minimize_whitespace(str(expr[0]))] = minimize_whitespace(
                str(expr[1].children[1])
            )

            # if the assigned value is a function result, save it to func_caller_dict
            if len(expr[1].children) > 3 and expr[1].children[3].type == syms.power:
                func_caller_dict[minimize_whitespace(str(expr[1].children[3].children[0]))] = expr[1]


    # propagate types from caller to callee (function)
    for child in flatten_some(lib2to3_node.children):
        decorators = None
        if child.type == syms.decorated:
            # skip decorators
            decorators = child.children[0]
            child = child.children[1]

        if child.type in (syms.async_stmt, syms.async_funcdef):
            # async def in 3.5 and 3.6
            child = child.children[1]

        if child.type != syms.funcdef:
            continue

        # funcdef: 'def' NAME parameters ['->' test] ':' suite
        func_name = minimize_whitespace(str(child.children[1]))
        if func_name in func_caller_dict:
            lineno = child.get_lineno()
            column = 1
            caller_expr = func_caller_dict[minimize_whitespace(str(func_name))]
            return_type = caller_expr.children[1]

            try:
                # TODO: condider class function - 'self' params
                propagate_parameters(child.children[2], caller_expr, var_anno_dict)
                propagate_returns(child.children, return_type)
            except ValueError as ve:
                raise ValueError(
                    f"Annotation problem in function {child.children[1]!r}: "
                    + f"{lineno}:{column}: {ve}"
                )


def propagate_returns(function, annotation):
    # funcdef: 'def' NAME parameters ['->' test] ':' suite
    if function[3] == _rarrow:
        function[4] = annotation
    elif function[3] == _colon:
        function.insert(3, new(_rarrow))
        function.insert(4, annotation)
    else:
        raise NotImplementedError(f"unexpected return token: {str(function[3])!r}")


def propagate_parameters(parameters, caller_expr, var_anno_dict):
    if isinstance(caller_expr.children[3].children[1].children[1], Leaf):
        # single param
        caller_params = [caller_expr.children[3].children[1].children[1]]
    else:
        caller_params = caller_expr.children[3].children[1].children[1].children

    # parameters: '(' [typedargslist] ')'
    # typedargslist: ((tfpdef ['=' test] ',')*
    #            ('*' [tname] (',' tname ['=' test])* [',' ['**' tname [',']]] | '**' tname [','])
    #            | tfpdef ['=' test] (',' tfpdef ['=' test])* [','])
    params = parameters.children[1:-1]
    if len(params) == 0:
        return
    elif len(params) > 1:
        raise NotImplementedError(f"unknown AST structure in parameters: {params}")

    # Simplify the possible data structures so we can just pull from it.
    if params[0].type == syms.typedargslist:
        params = params[0].children

    # TODO: consider args with defaults, varargs, kwonlyargs and kwarg
    typedargslist = []
    typedargslist.extend(gen_annotated_params(params, caller_params, var_anno_dict))

    if typedargslist:
        typedargslist = typedargslist[1:]  # drop the initial comma

        if len(typedargslist) == 1:
            # don't pack a single argument to be consistent with how lib2to3
            # parses existing code.
            body = typedargslist[0]
        else:
            body = Node(syms.typedargslist, typedargslist)
        parameters.children = [
            parameters.children[0],  # (
            body,
            parameters.children[-1],  # )
        ]
    else:
        parameters.children = [
            parameters.children[0],  # (
            parameters.children[-1],  # )
        ]


def gen_annotated_params(params, caller_params, var_anno_dict):
    idx = 0
    while params:
        yield new(_comma)

        param, default = pop_param(params)

        if param in (_star, _dstar):
            # unexpected *args, keyword-only args, or **kwargs
            raise ValueError(f"Unexpected *args, keyword-only args, or **kwargs")

        caller_param = caller_params[idx*2]
        if isinstance(caller_param, Leaf):
            caller_param_str = minimize_whitespace(str(caller_param))
        else:
            # caller_param is a kwarg
            caller_param_str = minimize_whitespace(str(caller_param.children[2]))

        annotation = var_anno_dict[caller_param_str]
        node = get_annotated_param(param, annotation)
        yield node
        if default:
            whitespace = " " if node.type == syms.tname else "" # tname: NAME [':' test]
            yield new(_eq, prefix=whitespace)
            yield new(default, prefix=whitespace)

        idx += 1


def get_annotated_param(node, annotation):
    if node.type == token.NAME:
        param_name = minimize_whitespace(str(node))
    elif node.type == syms.tname:
        param_name = minimize_whitespace(str(node.children[0]))
    else:
        raise NotImplementedError(f"unexpected node token: `{node}`")

    # annotation = var_anno_dict[param_name]
    ann_node = Leaf(token.NAME, annotation)
    ann_node.prefix = " "

    if node.type == token.NAME:
        return Node(syms.tname, [new(node), new(_colon), ann_node])
    else:
        # node.type == syms.tname, ignore exsisting annotation
        return Node(syms.tname, [new(node.children[0]), new(_colon), ann_node])


def pop_param(params):
    """ (Use this func when params is not empty)
    Pops the parameter and the "remainder" (comma, default value).

    Returns a tuple of ('name', default) or (_star, 'name') or (_dstar, 'name').
    """
    default = None

    name = params.pop(0)
    if name in (_star, _dstar):
        default = params.pop(0)
        if default == _comma:
            return name, default

    if params:
        remainder = params.pop(0)
        if remainder == _eq:
            default = params.pop(0)
            if params:
                remainder = params.pop(0)
        if remainder != _comma:
            raise ValueError(f"unexpected token: {remainder}")

    return name, default


def lib2to3_parse(src_txt):
    """Given a string with source, return the lib2to3 Node."""
    grammar = pygram.python_grammar_no_print_statement
    drv = driver.Driver(grammar, pytree.convert)
    if src_txt[-1] != "\n":
        nl = "\r\n" if "\r\n" in src_txt[:1024] else "\n"
        src_txt += nl
    try:
        result = drv.parse_string(src_txt, True)
    except ParseError as pe:
        lineno, column = pe.context[1]
        lines = src_txt.splitlines()
        try:
            faulty_line = lines[lineno - 1]
        except IndexError:
            faulty_line = "<line number missing in source>"
        raise ValueError(f"Cannot parse: {lineno}:{column}: {faulty_line}") from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])

    return result


def lib2to3_unparse(node):
    """Given a lib2to3 node, return its string representation."""
    code = str(node)
    return code


def flatten_some(children):
    """Generates nodes or leaves, unpacking bodies of try:except:finally: statements."""
    for node in children:
        if node.type in (syms.try_stmt, syms.suite):
            yield from flatten_some(node.children)
        else:
            yield node


def new(n, prefix=None):
    """lib2to3's AST requires unique objects as children."""

    if isinstance(n, Leaf):
        return Leaf(n.type, n.value, prefix=n.prefix if prefix is None else prefix)

    # this is hacky, we assume complex nodes are just being reused once from the
    # original AST.
    n.parent = None
    if prefix is not None:
        n.prefix = prefix
    return n


def minimize_whitespace(text):
    return re.sub(r"[\n\t ]+", " ", text, re.MULTILINE).strip()


_as = Leaf(token.NAME, "as", prefix=" ")
_colon = Leaf(token.COLON, ":")
_comma = Leaf(token.COMMA, ",")
_dot = Leaf(token.DOT, ".")
_dstar = Leaf(token.DOUBLESTAR, "**")
_eq = Leaf(token.EQUAL, "=", prefix=" ")
_lpar = Leaf(token.LPAR, "(")
_lsqb = Leaf(token.LSQB, "[")
_newline = Leaf(token.NEWLINE, "\n")
_none = Leaf(token.NAME, "None")
_rarrow = Leaf(token.RARROW, "->", prefix=" ")
_rpar = Leaf(token.RPAR, ")")
_rsqb = Leaf(token.RSQB, "]")
_star = Leaf(token.STAR, "*")
_ellipsis = Node(syms.atom, children=[new(_dot), new(_dot), new(_dot)])


if __name__ == "__main__":
    propagate(sys.argv[1])
