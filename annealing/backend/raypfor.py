import typed_ast.ast3 as ast
from typed_astunparse import unparse
import util.error as err
import util.asttools as asttools
import typesys.mytypes as mytypes
import typesys.tools as tytools
import libbase.entry as libkb
from midend.polyhedral import str_iterable # Todo: move to util

scheduling_policy = None
enable_cupy = False
default_num_processes = 2 # Todo: optionize?

def transform(tree, opts):
    global scheduling_policy, enable_cupy
    scheduling_policy = opts['ray_pfor_schedule']
    enable_cupy = opts['ray_pfor_cupy']
    RayPforCodegen().visit(tree)

# Todo: merge with midend.polyhedral.Replacer
class Replacer(ast.NodeTransformer):
    def __init__(self, targets, varmap, varmap_inner = {}, ufuncs = {}):
        self.targets = set(targets)
        self.varmap = varmap
        self.varmap_inner = varmap_inner
        self.ufuncs = ufuncs

    def visit_Subscript(self, N):
        if 'subscript' in self.targets:
            use_inner = isinstance(N.value, ast.Name) and N.value.id in self.varmap_inner
            varmap2 = self.varmap_inner[N.value.id] if use_inner else None

            N.value = self.visit(N.value)
            if use_inner:
                org = self.varmap
                self.varmap = varmap2
                N.slice = self.visit(N.slice)
                self.varmap = org
            else:
                N.slice = self.visit(N.slice)
        else:
            N = self.generic_visit(N)
        return N

    def visit_Call(self, N):
        if 'call' in self.targets:
            if isinstance(N.func, ast.Name):
                if N.func.id in self.ufuncs:
                    err.verbose(4, 'Numpy-to-cupy for user-defined function:', N.func.id)
                    uf = self.ufuncs[N.func.id]
                    for s in uf.body:
                        self.visit(s)
            else:
                assert 'name' not in self.targets
                self.targets.add('name')
                N.func = self.generic_visit(N.func)
                self.targets.remove('name')

            N.args = [self.visit_Call(item) if isinstance(item, ast.Call) else self.generic_visit(item) for item in N.args]
            N.keywords = [self.visit_Call(item) if isinstance(item, ast.Call) else self.generic_visit(item) for item in N.keywords]
        else:
            N = self.generic_visit(N)
        return N

    def visit_Name(self, N):
        if 'name' in self.targets:
            if N.id in self.varmap:
                N.id = self.varmap[N.id]
        else:
            N = self.generic_visit(N)
        return N

class RayPforCodegen(ast.NodeVisitor):

    def __init__(self):
        self.pfor_nodes = {}

    def pfor_template(self, policy, nproc, index, task, start, end, step, ins, out, dims):
        global scheduling_policy
        err.verbose(3, 'Scheduling policy:', scheduling_policy)
        if scheduling_policy == 'cyclic':
            return ('''
__nproc = os.environ.get('PYDD_NUM_THREADS')
__nproc = int(__nproc) if __nproc else %s
__count = ((%s-%s)//%s)
__div = __count // __nproc
__mod = __count %s __nproc
__status = []
for %s in range(__nproc):
    __chunk = __div + (1 if %s < __mod else 0)
    __s = %s.remote(%s, %s, __nproc, __chunk)
    __status.append(__s)

# Needed for performance?
# __status_ready, __status_pending = ray.wait(__status, num_returns = len(__status))

__status = ray.get(__status)
for %s in range(__nproc):
    __s = __status.pop(0)
    %s[%s+%s*%s : %s : 1*__nproc%s] = __s
''' %(nproc,
      end, start, step,
      '%',
      index,
      index,
      task, ins, index,
      index,
      out, start, step, index, end, dims))

        else:  # scheduling_policy == 'block'
            return ('''
__nproc = os.environ.get('PYDD_NUM_THREADS')
__nproc = int(__nproc) if  __nproc else %s
__step = -(-((%s-%s)//%s) // __nproc)
__status = []
for %s in range(%s, %s, __step):
    __chunk = __step if %s + __step <= %s else %s - %s
    __s = %s.remote(%s, %s, __chunk)
    __status.append(__s)

# Needed for performance?
# __status_ready, __status_pending = ray.wait(__status, num_returns = len(__status))

__status = ray.get(__status)
for %s in range(%s, %s, __step):
    __chunk = __step if %s + __step <= %s else %s - %s
    __s = __status.pop(0)
    %s[%s:%s+__chunk:%s%s] = __s
''' %(nproc,
      end, start, step,
      index, start, end,
      index, end, end, index,
      task, ins, index,
      index, start, end,
      index, end, end, index,
      out, index, index, step, dims))

    def codegen_pfor(self, N):
        assert isinstance(N, ast.For) and hasattr(N, 'pfor_info')
        if len(N.pfor_info.output) != 1:
            err.error('Ray pfor currently assumes one output')

        task = '__task' + str(len(self.pfor_nodes))
        self.pfor_nodes[N] = task
        err.verbose(3, 'Found pfor as function:', task)

        global default_num_processes
        nproc = default_num_processes
        index = N.pfor_info.index
        start = N.pfor_info.start
        end = N.pfor_info.end
        step = N.pfor_info.step
        ins = str_iterable(N.pfor_info.input)
        out = next(iter(N.pfor_info.output.keys()))
        out_type = N.pfor_info.output[out]
        assert isinstance(out_type, mytypes.ArrayType) and out_type.ndim != -1
        dims = ', :' * (out_type.ndim - 1)

        code = self.pfor_template(0, nproc, index, task, start, end, step, ins, out, dims)
        nodes = ast.parse(code)
        return nodes.body

    def visit_body(self, body):
        new_stmts = []
        transformed = False
        for stmt in body:
            if isinstance(stmt, ast.For) and hasattr(stmt, 'pfor_info'):
                stmts = self.codegen_pfor(stmt)
                new_stmts.extend(stmts)
                transformed = True
            else:
                self.visit(stmt)
                new_stmts.append(stmt)

        if transformed:
            body.clear()
            body.extend(new_stmts)

    def codegen_init(self, add_cupy):
        code = 'import cupy\n' if add_cupy else ''
        code += 'import os\nimport ray\nif not ray.is_initialized(): ray.init()'
        nodes = ast.parse(code)
        return nodes.body

    def codegen_ray_task(self, N, task_name):
        assert isinstance(N, ast.For) and hasattr(N, 'pfor_info')
        pfor = N.pfor_info
        N.type_comment = None

        idxvar = '__i'
        idxvar2 = '__i2'
        retvar = '__ret'
        chkvar = '__chunk'
        npvar = '__nproc'
        assert len(pfor.output) == 1
        out = next(iter(pfor.output.keys()))
        out_type = pfor.output[out]
        assert isinstance(out_type, mytypes.ArrayType) and out_type.shape

        global scheduling_policy
        etc = (npvar, chkvar) if scheduling_policy == 'cyclic' else (chkvar,)
        ins = pfor.input + (pfor.index,) + etc
        args0 = [ast.arg(arg = str(i), annotation = None, type_comment = None) for i in ins]
        args = ast.arguments(args0, vararg = None, kwonlyargs = [], kw_defaults = [], kwarg = None, defaults = [])

        # Insert '__ret = zeros(...)' before N
        shape = (chkvar,) + out_type.shape[1:]
        rettype = tytools.array_with_check(out_type.etype, out_type.ndim, shape, out_type.module)
        init = asttools.init_array_statement(retvar, rettype)

        # Change to 'for idxvar in ...'
        N.target = asttools.name(idxvar, True)
        assert isinstance(N.iter, ast.Call) and isinstance(N.iter.func, ast.Name) and N.iter.func.id == 'range'
        N_pfor_index = asttools.name(pfor.index)
        N_pfor_step = asttools.name(pfor.step)
        if scheduling_policy == 'cyclic':
            # Change to 'range(pfor.start + pfor.step * pfor.index, pfor.end, pfor.step * npvar)'
            N_pfor_start = asttools.name(pfor.start)
            N_pfor_end = asttools.name(pfor.end)
            v = ast.BinOp(left = N_pfor_step, op = ast.Mult(), right = N_pfor_index)
            N_start = ast.BinOp(left = N_pfor_start, op = ast.Add(), right = v)
            N_step = ast.BinOp(left = N_pfor_step, op = ast.Mult(), right = asttools.name(npvar))
            N.iter.args = [N_start, N_pfor_end, N_step]
        else:
            # Change to 'range(pfor.index, pfor.index + chkvar, pfor.step)'
            N_end = ast.BinOp(left = N_pfor_index, op = ast.Add(), right = asttools.name(chkvar))
            N.iter.args = [N_pfor_index, N_end, N_pfor_step]

        # Replace index and pfor.out in N.body
        varmap = {pfor.index: idxvar, out: retvar}
        varmap_inner = {out: {pfor.index: idxvar2}}
        rep = Replacer(('subscript', 'name'), varmap, varmap_inner)
        for s in N.body:
            rep.visit(s)

        # Insert 'idxvar2 = 0' before N
        tar = asttools.name(idxvar2, True)
        init2 = ast.Assign(targets = [tar], value = ast.Num(n = 0), type_comment = None)

        # Append 'idxvar2 += 1' to N.body
        incr = ast.AugAssign(target = tar, op = ast.Add(), value = ast.Num(n = 1))
        N.body.append(incr)

        ret = ast.Return(value = asttools.name(retvar))

        rr = ast.Attribute(value = asttools.name('ray'), attr = 'remote', ctx = ast.Load())
        global enable_cupy
        if enable_cupy:
            rrf = rr
            rrk = ast.keyword(arg = 'num_gpus', value = ast.Num(n = 1))
            rr = ast.Call(func = rrf, args = [], keywords = [rrk])

        func = ast.FunctionDef(name = task_name, args = args, body = [init, init2, N, ret],
                               decorator_list = [rr], returns = None, type_comment = None)
        return func

    def codegen_cupy_transfer(self, arr, cpvar, to_numpy = False):
        fname = 'asnumpy' if to_numpy else 'asarray'
        err.verbose(4, 'Transfer: %s = %s(%s)' %(arr, fname, arr))
        src = asttools.name(arr)
        cp = asttools.name(cpvar)
        func = ast.Call(func = ast.Attribute(value = cp, attr = fname), args = [src], keywords = [])

        tar = asttools.name(arr, True)
        transfer = ast.Assign(targets = [tar], value = func, type_comment = None)
        err.verbose(5, 'Transfer (ast):', ast.dump(transfer))
        return transfer

    def convert_to_cupy(self, task, pfor, N_module):
        cpvar = libkb.module_to_var('cupy')
        assert len(pfor.output) == 1 and cpvar

        # WORKING: Test implementation
        varmap = {npvar: cpvar for npvar in libkb.module_to_var('numpy', True)}
        ufs = {f.name: f for f in N_module.body if isinstance(f, ast.FunctionDef)}
        rep = Replacer(('call',), varmap, ufuncs = ufs)
        for s in task.body:
            rep.visit(s)

        h2ds = []
        for arr in pfor.intypes:
            aty = pfor.intypes[arr]
            if isinstance(aty, mytypes.ArrayType) and aty.module != 'cupy':
                h2ds.append(self.codegen_cupy_transfer(arr, cpvar))
        task.body = h2ds + task.body

        oty = next(iter(pfor.output.values()))
        if isinstance(oty, mytypes.ArrayType) and oty.module != 'cupy':
            retvar = '__ret'
            d2h = self.codegen_cupy_transfer(retvar, cpvar, True)
            assert isinstance(task.body[-1], ast.Return)
            task.body.insert(-1, d2h)


### AST visitors ###

    def visit_Module(self, N):
        global enable_cupy
        add_cupy = enable_cupy and not libkb.module_to_var('cupy')
        if add_cupy:
            libkb.import_module('cupy', 'cupy')

        self.generic_visit(N)

        if self.pfor_nodes:
            inits = self.codegen_init(add_cupy)
            tasks = []
            for p in self.pfor_nodes:
                t = self.codegen_ray_task(p, self.pfor_nodes[p])
                tasks.append(t)
                if enable_cupy:
                    self.convert_to_cupy(t, p.pfor_info, N)

            first = next((f for f in N.body if isinstance(f, ast.FunctionDef)), None)
            assert first
            pos = N.body.index(first)

            new_stmts = N.body[:pos] + inits + tasks + N.body[pos:]
            N.body = new_stmts

    def visit_FunctionDef(self, N):
        err.verbose(2, 'Ray pfor codegen:', N.name)

        self.visit_body(N.body)

    def visit_If(self, N):
        self.visit_body(N.body)
        self.visit_body(N.orelse)

    def visit_While(self, N):
        self.visit_body(N.body)
        self.visit_body(N.orelse)

    def visit_For(self, N):
        self.visit_body(N.body)
        self.visit_body(N.orelse)
