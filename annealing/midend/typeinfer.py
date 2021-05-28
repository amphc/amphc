import typed_ast.ast3 as ast
from typed_astunparse import unparse
import inspect
import util.error as err
import util.vartools as vartools
import util.asttools as asttools
import typesys.rules as rules
import typesys.mytypes as mytypes
import typesys.tools as tytools
import libbase.tools as libtools
import libbase.entry as libkb

is_python_input = False
fuse_topfunc = False

def infer(tree, opts):
    global is_python_input, fuse_topfunc
    is_python_input = opts['python_input']
    fuse_topfunc = opts['topfunc_fusion']
    rules.presume_const_annot = opts['const_annot']

    PreProcessor().visit(tree)
    TypeInference().visit(tree)

'''
Preprocess to split:
    A = allocate(size_expr1, size_expr2, ..., size_exprN)
into:
    __var1 = size_expr1
    __var2 = size_expr2
    ...
    __varN = size_exprN
    A = allocate(__var1, __var2, ..., __varN)

1) allocate function contains empty, zeros, and etc.
2) size_expr is expression such as BinOp of "n * m"
'''
class PreProcessor(ast.NodeTransformer):
    def __init__(self):
        self.search_expr = False
        self.split_exprs = {}
        self.vars = set() # WORKING: fill with existing variables

    def is_alloc_with_size(self, N):
        # WORKING: generalize condition by libkb
        return isinstance(N, ast.Call) and isinstance(N.func, ast.Attribute) and N.func.attr == 'zeros'

    def split_alloc_stmt(self, N):
        self.search_expr = True
        N.value = self.visit(N.value)
        self.search_expr = False

        if self.split_exprs:
            stmts = [asttools.assign(var, self.split_exprs[var]) for var in self.split_exprs]
            stmts.append(N)
            self.split_exprs.clear()
        else:
            stmts = [N]

        return stmts

    def visit_body(self, body):
        new_body = []
        for stmt in body:
            if isinstance(stmt, ast.Assign) and self.is_alloc_with_size(stmt.value):
                stmts = self.split_alloc_stmt(stmt)
            else:
                stmts = []

            if len(stmts) >= 2:
                new_body.extend(stmts)
            else:
                new_body.append(self.visit(stmt))

        return new_body


### AST visitors ###

    def visit_FunctionDef(self, N):
        N.body = self.visit_body(N.body)
        return N

    def visit_If(self, N):
        N.body = self.visit_body(N.body)
        N.orelse = self.visit_body(N.orelse)
        return N

    def visit_While(self, N):
        return self.visit_If(N)

    def visit_For(self, N):
        return self.visit_If(N)

    def visit_BinOp(self, N):
        if self.search_expr:
            var = vartools.new_var(self.vars)
            self.split_exprs[var] = N
            return asttools.name(var)
        else:
            N.left = self.visit(N.left)
            N.right = self.visit(N.right)
            return N


class TypeInference(ast.NodeVisitor):
    def __init__(self):
        # Intrepydd program check
        self.num_modules = 0
        self.in_func_body = False

        # Function signatures with type annotations
        self.signatures = {}

    def collect_signature(self, func_name, args, defaults, returns):
        n = len(args)
        nnd = n - len(defaults)
        assert nnd >= 0

        params = []
        for i in range(n):
            p = args[i].arg
            ty = rules.annotation(args[i].annotation)
            assert p and ty
            if i < nnd:
                dty = None
            else:
                d = defaults[i-nnd]
                self.visit(d)
                dty = d.type
            params.append((p, ty, dty))
            err.verbose(3, p, '=>', ty, '/ default:', dty)

        rtype = rules.annotation(returns) if returns else None
        err.verbose(3, 'return =>', rtype)

        self.signatures[func_name] = tytools.TypedSignature(func_name, params, rtype)

    def collect_imported_module(self, names):
        for name in names:
            module = name.name
            tar = name.asname if name.asname else module
            sts = libkb.import_module(module, tar)
            if sts: err.verbose(4, 'Import:', module, 'as:', tar)

    def fuse_funcs(self, stmt1, stmt2, body_pydd):
        if stmt1:
            name1, name2 = stmt1.value.func.id, stmt2.value.func.id
            err.verbose(2, 'Fusing functions:', name1, 'and', name2)
            sargs1, sargs2 = stmt1.value.args, stmt2.value.args
            fdef1 = next(f for f in body_pydd if isinstance(f, ast.FunctionDef) and f.name == name1)
            fdef2 = next(f for f in body_pydd if isinstance(f, ast.FunctionDef) and f.name == name2)
            fargs1, fargs2 = fdef1.args.args, fdef2.args.args

            pos = 0
            nargs2 = set([n.id for n in sargs2])
            for i in range(len(sargs1)):
                if sargs1[i].id not in nargs2:
                    sargs2.insert(pos, sargs1[i])
                    fargs2.insert(pos, fargs1[i])
                    pos += 1

            sret1 = stmt1.targets[0] if isinstance(stmt1, ast.Assign) else stmt1.target
            assert isinstance(sret1, ast.Name)
            for i in range(len(sargs2)):
                if sargs2[i].id == sret1.id:
                    sargs2.pop(i)
                    fargs2.pop(i)
                    break

            for s in fdef1.body:
                if isinstance(s, ast.Return):
                    assert isinstance(s.value, ast.Name) and s.value.id == sret1.id
                    fdef1.body.remove(s)
                    break

            b = fdef1.body + fdef2.body
            fdef2.body.clear()
            fdef2.body.extend(b)
            body_pydd.remove(fdef1)  # WORKING: check if still used

            err.verbose(3, 'Collect fused function signatures:', fdef2.name)
            self.collect_signature(fdef2.name, fdef2.args.args, fdef2.args.defaults, fdef2.returns)

        return stmt2

    def fuse_top_function(self, body_pydd, body_py):
        heavy_funcs = set()
        for sig in self.signatures.values():
            def is_mdarr_type(aty): return isinstance(aty, mytypes.ArrayType) and aty.ndim >= 2
            mdarr_param = next((p for p in sig.params if is_mdarr_type(p[1])), None)
            if mdarr_param and is_mdarr_type(sig.returns):
                heavy_funcs.add(sig.name)

        new_body_py = []
        heavy_stmt = None
        for stmt in body_py:
            fusable = False
            if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id in heavy_funcs):
                args = stmt.value.args
                params = self.signatures[stmt.value.func.id].params
                n = len(args)
                if len(params) == n:
                    def same_name(a, p): return isinstance(a, ast.Name) and a.id == p[0]
                    if not next((x for x in zip(args, params) if not same_name(x[0], x[1])), None):
                        # WORKING: check returns
                        fusable = True
            if fusable:
                heavy_stmt = self.fuse_funcs(heavy_stmt, stmt, body_pydd)
            else:
                if heavy_stmt:
                    new_body_py.append(heavy_stmt)
                    heavy_stmt = None
                new_body_py.append(stmt)
        if heavy_stmt:
            new_body_py.append(heavy_stmt)

        body_py.clear()
        body_py.extend(new_body_py)

    def split_main_and_kernel(self, N):
        body_pydd, body_py = [], []
        for stmt in N.body:
            if isinstance(stmt, ast.Import):
                body_pydd.append(stmt)
                body_py.append(stmt)
            elif isinstance(stmt, ast.FunctionDef):
                body_pydd.append(stmt)
            else:
                body_py.append(stmt)

        global fuse_topfunc
        if fuse_topfunc:
            err.verbose(2, 'Fuse top functions')
            self.fuse_top_function(body_pydd, body_py)
        N.body = body_pydd
        N.body_py = body_py

    def collect_attribute_call_info(self, N):
        name = N.attr
        value = N.value
        if name in value.type.attributes:
            mname = None
            arg0 = value
        elif isinstance(value, ast.Name):
            m = libkb.var_to_module(value.id)
            if not m:
                err.error('Unsupported or not-imported module:', value.id, 'for attribute call')
            mname = [m]
            arg0 = None
        elif isinstance(value, ast.Attribute):
            n, m, a = self.collect_attribute_call_info(value)
            assert not a
            mname = m + [n]
            arg0 = None
        else:
            err.error('Unsupported attribute call:', unparse(N))

        return name, mname, arg0

    def assign_type(self, tar, ty):
        if isinstance(tar, ast.Name):
            tar.type = ty
            self.curr_type_table.add(tar.id, ty)
            err.verbose(3, tar.id, '=>', ty)
        elif isinstance(tar, ast.Tuple):
            n = len(tar.elts)
            assert isinstance(ty, tuple) and len(ty) == n
            tar.type = ty
            for i in range(n):
                self.assign_type(tar.elts[i], ty[i])
        elif isinstance(tar, ast.Subscript):
            self.visit(tar)
            if tar.type == ty:
                err.verbose(3, tar.value.id, '[...] =>', ty)
            else:
                err.warning('Inconsistent types between left:', ty, 'and right:', tar.type, 'for subscript', unparse(tar))
        else:
            err.error('Unsupported type assignment target:', unparse(tar))

    def set_type_and_df(self, N, info):
        if isinstance(info, mytypes.MyType):
            N.type = info
        else:
            assert len(info) == 2
            N.type, N.func_df = info

    def handle_branch(self, N):
        ptbl = self.curr_type_table
        ptbl.fork()

        self.curr_type_table = ptbl.body
        for stmt in N.body:
            self.visit(stmt)

        err.verbose(3, 'ELSE')
        self.curr_type_table = ptbl.orelse
        for stmt in N.orelse:
            self.visit(stmt)

        ptbl.join()
        self.curr_type_table = ptbl
        err.verbose(3, 'Joined types:', self.curr_type_table)


### AST visitors ###

    def visit(self, N):
        if not self.in_func_body:
            okay = [ast.Module, ast.FunctionDef, ast.Import, ast.alias, ast.ImportFrom]
            cl = type(N)
            if cl not in okay:
                err.error('Illegal out-of-function statement/expression:', unparse(N))
        ast.NodeVisitor.visit(self, N)

    def visit_Module(self, N):
        self.num_modules += 1
        if self.num_modules > 1:
            err.error('Multiple modules are not supported.')

        for stmt in N.body:
            if isinstance(stmt, ast.FunctionDef):
                err.verbose(2, 'Collect function signatures:', stmt.name)
                self.collect_signature(stmt.name, stmt.args.args, stmt.args.defaults, stmt.returns)
            elif isinstance(stmt, ast.Import):
                err.verbose(2, 'Collect imported modules:', [n.name for n in stmt.names])
                self.collect_imported_module(stmt.names)

        global is_python_input
        if is_python_input:
            err.verbose(2, 'Split main and kernel')
            self.split_main_and_kernel(N)

        self.generic_visit(N)

    def visit_FunctionDef(self, N):
        err.verbose(2, 'Type infer function:', N.name)
        self.in_func_body = True

        mysig = self.signatures[N.name]
        types = {tar: mytypes.AnyType() for tar in libkb.v2m}
        types.update(mysig.params_as_dict())
        self.curr_type_table = tytools.TypeTable(types)
        self.curr_ret_type = mysig.returns

        for stmt in N.body:
            self.visit(stmt)

        N.signature = mysig
        N.type_table = self.curr_type_table
        err.verbose(2, N.name, ':', N.type_table)
        self.in_func_body = False

    def visit_Assign(self, N):
        self.visit(N.value)
        ty = N.value.type
        assert ty

        n = len(N.targets)
        if n == 1:
            self.assign_type(N.targets[0], ty)
        else:
            m = len(ty) if isinstance(ty, tuple) else -1
            if m != n:
                err.error('Type inference failed; # type annotations =', m, 'doesn\'t match with # defined variables =' + n)
            for i in range(n):
                self.assign_type(N.targets[i], ty[i])

    def visit_AugAssign(self, N):
        self.visit(N.target)
        self.visit(N.value)

        ret = rules.operator(N.target.type, N.value.type, N.op)
        self.set_type_and_df(N, ret)

        assert N.type
        self.assign_type(N.target, N.type)

    def visit_AnnAssign(self, N):
        if N.value:
            self.visit(N.value)
            ty1 = N.value.type
        else:
            self.visit(N.target)
            ty1 = N.target.type

        ty2 = rules.annotation(N.annotation)
        assert ty1 and ty2
        if not tytools.type_including(ty1, ty2):
            err.warning('Type annotation:', ty2, 'doesn\'t match with inferred type', ty1, 'for variable:', unparse(N.target))
        self.assign_type(N.target, ty2)

    def visit_Return(self, N):
        if N.value:
            self.visit(N.value)
            ty = N.value.type
            assert ty
            if not self.curr_ret_type or not tytools.type_including(ty, self.curr_ret_type):
                err.warning('Return type annotation:', self.curr_ret_type, 'doesn\'t match with inferred type:', ty)

    def visit_If(self, N):
        err.verbose(3, 'START IF')
        self.handle_branch(N)
        err.verbose(3, 'END IF')

    def visit_While(self, N):
        err.verbose(3, 'START WHILE')
        self.handle_branch(N)  # WORKING: handle types from previous iterations (same in visit_For)
        err.verbose(3, 'END WHILE')

    def visit_For(self, N):
        err.verbose(3, 'START FOR')
        self.visit(N.iter)
        ty = rules.iterator(N.iter.type)
        assert ty
        self.assign_type(N.target, ty)
        self.handle_branch(N)
        err.verbose(3, 'END FOR')

    def visit_BinOp(self, N):
        self.visit(N.left)
        self.visit(N.right)

        ret = rules.operator(N.left.type, N.right.type, N.op)
        self.set_type_and_df(N, ret)

    def visit_UnaryOp(self, N):
        self.visit(N.operand)

        ret = rules.unaryop(N.operand.type, N.op)
        self.set_type_and_df(N, ret)

    def visit_BoolOp(self, N):
        for arg in N.values:
            self.visit(arg)
        arg_types = [arg.type for arg in N.values]

        ret = rules.boolop(arg_types, N.op)
        self.set_type_and_df(N, ret)

    def visit_Compare(self, N):
        if len(N.comparators) != len(N.ops):
            err.error('# comparators must be same as # ops for compare expression:', unparse(N))

        self.visit(N.left)
        for arg in N.comparators:
            self.visit(arg)
        arg_types = [N.left.type] + [arg.type for arg in N.comparators]

        ret = rules.cmpop(arg_types, N.ops)
        self.set_type_and_df(N, ret)

    def visit_Expr(self, N):
        self.generic_visit(N)

    def is_type_required_call(self):
        curr = inspect.currentframe()
        callers = inspect.getouterframes(curr, 2)
        names = [c[3] for c in callers]
        nonreq = ['visit', 'generic_visit', 'visit_Expr']
        l = len(names)
        if l >= 5 and names[2:5] == nonreq:
            return False
        else:
            if l > 2:
                err.verbose(5, 'Outer callers:', names[2:min(5, l)])
            return True

    def visit_Call(self, N):
        for arg in N.args:
            self.visit(arg)
        for key in N.keywords:
            self.visit(key)

        if isinstance(N.func, ast.Name):
            N.func.type = mytypes.AnyType() # Special type handling for ast.Name of N.func
            name = N.func.id
            mname = None
            args = N.args
        elif isinstance(N.func, ast.Attribute):
            self.visit(N.func)
            name, mname, arg0 = self.collect_attribute_call_info(N.func)
            args = [arg0] + N.args if arg0 else N.args
        else:
            err.error('Unsupported call:', unparse(N.func))

        arg_types = [arg.type for arg in args]
        key_types = {key.arg: key.value.type for key in N.keywords}
        keywords = {key.arg: key.value for key in N.keywords}

        if name in self.signatures:
            self.signatures[name].verify_args(arg_types, key_types)
            ty = self.signatures[name].returns
            if not ty and self.is_type_required_call():
                err.error('Type annotation is missing for function:', name)
            df = libtools.default_dfinfo
            self.set_type_and_df(N, (ty, df)) # Todo: IPA to compute user function dataflow
        else:
            ret = rules.call(name, arg_types, key_types, args, keywords, mname)
            self.set_type_and_df(N, ret)

    '''
    Currently assumed usage of attribute:
     - Case 1: call function from imported module, e.g., a = cupy.fft.ifft(b)
     - Case 2: specify dtype, e.g., b = cupy.zeros((10,20), dtype = numpy.complex128)
    They are handled at visit_Call via module table while assigning AnyType to attribute node.
    '''
    def visit_Attribute(self, N):
        self.visit(N.value)

        # Todo: generalize via libkb
        if isinstance(N.value.type, mytypes.ArrayType) and N.attr == 'T':
            arg = N.value
            ret = rules.call('transpose', [arg.type], {}, [arg], {}, None)
            self.set_type_and_df(N, ret)
        elif isinstance(N.value, ast.Name):
            mdinfo = libkb.var_to_module(N.value.id, True)
            if mdinfo and hasattr(mdinfo, 'const') and N.attr in mdinfo.const:
                N.type = mdinfo.const[N.attr]
            else:
                N.type = mytypes.AnyType()
        else:
            N.type = mytypes.AnyType()

    '''
    Current Intrepydd semantics: ast.Tuple can appear only in:
     - Left hand of assignment -- Assign.targets[i]
     - Right hand of assignment -- Assign.value
     - Function argument -- Call.args[i]
     - Subscript access -- Subscript.value
    Example:
     (x, y) = get_2elms_tuple()
     t = (calc_scalar(x), calc_scalar(y))
     use_tuple(t)
     v = t[0] * t[1]
    '''
    def visit_Tuple(self, N):
        for elm in N.elts:
            self.visit(elm)
        ty = tuple(map(lambda x: x.type, N.elts))
        assert all(ty)
        # Todo: Merge dfs, tuple(map(lambda x: x.func_df if hasattr(x, 'func_df') else None))
        N.type = ty

    def visit_Subscript(self, N):
        self.visit(N.value)
        self.visit(N.slice)
        N.type = rules.subscript(N.value.type, N.slice)

    # def visit_ExtSlice(self, N): => same as ast.NodeVisitor

    def visit_Slice(self, N):
        for val in (N.lower, N.upper, N.step):
            if val:
                self.visit(val)
                if not isinstance(val.type, mytypes.IntType):
                    err.error('Slice must be int type, but found:', unparse(N))

    def visit_Index(self, N):
        self.visit(N.value)
        if not isinstance(N.value.type, mytypes.IntType):
            err.error('Index must be int type, but found:', unparse(N))

    def visit_Name(self, N):
        strict_type = False  # Todo: optionize
        ty = self.curr_type_table.get(N.id)
        if not ty:
            err.error('Type inference failed; undefined type for:', N.id)
        elif strict_type and type(ty) in {mytypes.AnyType, mytypes.NumType}:
            err.error('Type inference failed; ambiguous type for:', N.id)
        N.type = ty

    def visit_Num(self, N):
        N.type = rules.num(N.n)

    def visit_NameConstant(self, N):
        N.type = rules.name_constant(N.value)

    def visit_Str(self, N):
        N.type = rules.string()
