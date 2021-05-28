import typed_ast.ast3 as ast
from typed_astunparse import unparse
import collections
import islpy as isl
import util.error as err
import util.asttools as asttools
import libbase.tools as libtools
import libbase.entry as libkb
import typesys.mytypes as mytypes

enable_layout_opt = False

def optimize(tree, opts):
    global enable_layout_opt
    enable_layout_opt = opts['poly_opt_layout']
    builder = ScopBuilder()
    builder.visit(tree)

    for N in builder.scops:
        scop = builder.scops[N]
        fname = N.name
        err.verbose(2, 'Polyhedral dependence analysis for:', fname)
        scop.dependence_analysis()
        err.verbose(2, 'Polyhedral transformations for:', fname)
        scop.transform()
        err.verbose(2, 'Polyhedral codegen for:', fname)
        scop.codegen(N)

def extract_loop_bounds(N):
    assert isinstance(N.target, ast.Name) # Todo: handle other than ast.Name
    tar = N.target.id

    it = N.iter
    assert isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == 'range'
    if len(it.args) == 1:
        start = 0
        end = asttools.extract_value(it.args[0])
        step = 1
    else:
        start = asttools.extract_value(it.args[0])
        end = asttools.extract_value(it.args[1])
        step = asttools.extract_value(it.args[2]) if len(it.args) == 3 else 1

    return (start, end, step, tar)

def str_iterable(it):
    xstr = ''
    for x in it:
        xstr = xstr + ', ' + str(x) if xstr else str(x)
    return xstr

def str_bounds(bounds):
    bstr = ''
    for b in bounds:
        if b[0] is not None:
            if b[2] is not None:
                bs = '%s <= %s < %s' % (b[0], b[1], b[2])
            else:
                bs= '%s <= %s' % (b[0], b[1])
        else:
            bs = '%s < %s' % (b[1], b[2]) if b[2] is not None else ''
        if bs:
            bstr = bstr + ' and ' + bs if bstr else bs
    return bstr

class StmtInfo:
    def __init__(self, id, ast_node, outer_loops,
                 indices_l, indices_a, bounds_l, bounds_a, writes, reads, schedule):
        self.id = id
        self.ast_node = ast_node
        self.outer_loops = outer_loops
        self.indices_l = indices_l
        self.indices_a = indices_a
        self.indices = indices_l + indices_a
        self.bounds_l = bounds_l
        self.bounds_a = bounds_a
        self.bounds = bounds_l + bounds_a
        self.writes = writes
        self.reads = reads
        self.schedule_org = schedule
        self.schedule_new = []

    def str_space(self):
        istr = str(self.id)
        xstr = str_iterable(self.indices_l + self.indices_a)
        return 'S%s[%s]' % (istr, xstr)

    def str_domain(self, globals = set()):
        gstr = str_iterable(globals)
        sstr = self.str_space()
        bstr = str_bounds(self.bounds_l + self.bounds_a)
        if gstr:
            return '[%s] -> {%s : %s}' % (gstr, sstr, bstr)
        else:
            return '{%s : %s}' % (sstr, bstr)

    def str_arrays(self, is_write = True, globals = set()):
        gstr = str_iterable(globals)
        if gstr:
            fstr = '[%s] -> {' % (gstr)
        else:
            fstr = '{'
        accesses = self.writes if is_write else self.reads
        return tuple(map(lambda a: '%s %s}' % (fstr, a.str_array()), accesses))

    def str_accesses(self, is_write = True, globals = set()):
        gstr = str_iterable(globals)
        sstr = self.str_space()
        if gstr:
            fstr = '[%s] -> {%s ->' % (gstr, sstr)
        else:
            fstr = '{%s ->' % sstr
        accesses = self.writes if is_write else self.reads
        return tuple(map(lambda a: '%s %s}' % (fstr, str(a)), accesses))

    def str_schedule(self, is_new = False, max_depth = -1):
        spstr = self.str_space()
        sche = self.schedule_org if not is_new else tuple(self.schedule_new)
        l = max_depth * 2 + 1 if max_depth != -1 else len(sche)
        ext = sche + (0,) * (l - len(sche))
        scstr = str_iterable(ext)
        return '{%s -> [%s]}' % (spstr, scstr)

    class AccessInfo:
        def __init__(self, array, atype, ast_node, indices, bounds):
            self.array = array
            self.atype = atype
            self.ast_node = ast_node
            self.indices = indices
            self.indices_new = ()
            self.bounds = bounds

        def __str__(self):
            astr = '%s[%s]' % (str(self.array), str_iterable(self.indices))
            bstr = str_bounds(self.bounds)
            if bstr:
                return '%s : %s' % (astr, bstr)
            else:
                return '%s' % (astr)

        def str_array(self):
            elms = tuple(map(lambda i: 'e' + str(i), range(len(self.indices))))
            return '%s[%s]' % (str(self.array), str_iterable(elms))

class ScopBuilder(ast.NodeVisitor):
    def __init__(self):
        self.scops = {} # key: FunctionDef object, value: Scop

    def is_scop_candidate_func(self, N):
        assert isinstance(N, ast.FunctionDef)

        # WORKING: test implementations for cost analysis
        first = next((s for s in N.body if isinstance(s, ast.For)), None)
        if not first:
            err.verbose(3, 'Not sufficient cost for function:', N.name)
            return False

        return True

    def init_status(self):
        self.curr_stmt_id = 0
        self.curr_any_id = 0
        self.max_depth = 0
        self.globals = set()
        self.anys = set()
        self.time_stamp = [0]
        self.outer_bounds = []
        self.outer_loops = []
        self.stmts_info = []

    def update_status_loop_start(self, N):
        idx = N.target.id
        self.time_stamp.append(idx)
        self.time_stamp.append(0)
        b = extract_loop_bounds(N)
        if b[2] != 1:
            err.error('Unsupported loop step for polyhedral compilation: step =', b[2])
        c = (b[0], b[3], b[1])
        self.outer_bounds.append(c)
        self.outer_loops.append(N)

    def update_status_loop_end(self):
        self.time_stamp.pop()
        self.time_stamp.pop()
        self.time_stamp[-1] += 1
        self.outer_bounds.pop()
        self.outer_loops.pop()

    def update_status_next_stmt(self):
        self.time_stamp[-1] += 1
        self.curr_stmt_id += 1

    def new_any_var(self):
        # Todo: check if this name is already used -> to util/asttools.py
        v = '__any_' + str(self.curr_any_id)
        self.curr_any_id += 1
        self.anys.add(v)
        return v

    def set_current_scop(self, N):
        assert len(self.time_stamp) == 1 and isinstance(self.time_stamp[0], int)
        params = self.globals | self.anys
        md = self.max_depth

        Sdoms, Adoms = [], []
        W, R, WR, O = None, None, None, None
        space_to_id, space_to_array = {}, {}
        for s in self.stmts_info:
            D_s = isl.Set(s.str_domain(params))
            Sdoms.append(D_s)
            space_to_id[D_s.get_space()] = s.id

            i = 0
            for astr in s.str_arrays(True, params):
                A_s = isl.Set(astr)
                sp = A_s.get_space()
                if sp not in space_to_array:
                    space_to_array[sp] = s.writes[i].array
                    Adoms.append(A_s)
                i += 1

            for wstr in s.str_accesses(True, params):
                W_s = isl.Map(wstr).intersect_domain(D_s)
                if W: W = W.add_map(W_s)
                else: W = isl.UnionMap(W_s)
                if WR: WR = WR.add_map(W_s)
                else:  WR = isl.UnionMap(W_s)

            for rstr in s.str_accesses(False, params):
                R_s = isl.Map(rstr).intersect_domain(D_s)
                if R: R = R.add_map(R_s)
                else: R = isl.UnionMap(R_s)
                if WR: WR = WR.add_map(R_s)
                else:  WR = isl.UnionMap(R_s)

            O_s = isl.Map(s.str_schedule(max_depth = md))
            if O: O = O.add_map(O_s)
            else: O = isl.UnionMap(O_s)

        scop = Scop(self.stmts_info, md, self.globals, self.anys, params,
                    Sdoms, Adoms, space_to_id, space_to_array, W, R, WR, O)
        self.scops[N] = scop

    def collect_args(self, N):
        args = None
        if isinstance(N, ast.BinOp):
            args = [N.left, N.right]
        elif isinstance(N, ast.UnaryOp):
            args = [N.operand]
        elif isinstance(N, ast.BoolOp):
            args = N.values
        elif isinstance(N, ast.Compare):
            args = [N.left] + N.comparators
        elif isinstance(N, ast.Call):
            func = N.func
            if isinstance(func, ast.Name):
                args = N.args
            elif isinstance(func, ast.Attribute):
                if func.attr in func.value.type.attributes:
                    args = [func.value] + N.args
                else:
                    args = N.args
            else:
                err.error('Unsupported call:', unparse(func))

        elif isinstance(N, ast.AugAssign):
            args = [N.target, N.value]

        return args

    def get_subscripted_var(self, N, subscr):
        if isinstance(N, ast.Subscript):
            if not isinstance(N.value, ast.Name):
                err.error('Unsupported variable expression:', unparse(N))
            dims = N.slice.dims if isinstance(N.slice, ast.ExtSlice) else [N.slice]
            if subscr == libtools.subscr_all and isinstance(N.type, mytypes.ArrayType):
                subscr = [libtools.all_in_dim] * N.type.ndim
            sub0 = list(subscr)
            sub1 = []
            for d in dims:
                if isinstance(d, ast.Index):
                    if not isinstance(d.value, ast.Name):
                        err.error('Unsupported subscript expression:', unparse(d.value))
                    sub1.append(d.value.id)
                else:
                    assert isinstance(d, ast.Slice)
                    sub1.append(sub0.pop(0))
            sub1 += sub0
            return (N.value.id, N.value.type, N, tuple(sub1))

        elif isinstance(N, ast.Name):
            return (N.id, N.type, N, subscr)
        elif isinstance(N, ast.Num):
            return None
        elif isinstance(N, ast.Attribute):
            # Todo: generalize via libkb
            if isinstance(N.value.type, mytypes.ArrayType) and N.attr == 'T':
                err.error('T (transpose) is not yet supported')
            elif isinstance(N.value, ast.Name):
                mdinfo = libkb.var_to_module(N.value.id, True)
                if mdinfo and hasattr(mdinfo, 'const') and N.attr in mdinfo.const:
                    return None
            err.error('Unsupported attribute expression:', unparse(N))
        else:
            err.error('Unsupported variable expression:', unparse(N))

    def extract_uses(self, N, out_subscr):
        unresolved = set()
        used = []

        args = self.collect_args(N)
        if args:
            df = N.func_df if hasattr(N, 'func_df') else libtools.default_dfinfo
            df2 = df.get_assigned(out_subscr)
            err.verbose(4, 'df-info:', df2, '/ assigning:', out_subscr)
            unresolved |= set(map(lambda i: out_subscr[i] if out_subscr[i] != df2.out_subscr[i]
                                  else None, range(len(out_subscr))))
            if len(unresolved) > 1 or len(unresolved) > 0 and None not in unresolved:
                err.verbose(4, 'unresolved:', unresolved)

            for i in range(len(args)):
                sub = df2.in_subscr[i] if isinstance(df2.in_subscr, list) else df2.in_subscr
                ret = self.extract_uses(args[i], sub)
                unresolved |= ret[0]
                used += ret[1]

        elif isinstance(N, ast.Tuple):
            for elt in N.elts:
                # Todo: Handle exact dataflow for tuple
                ret = self.extract_uses(elt, out_subscr)
                unresolved |= ret[0]
                used += ret[1]

        else:  # leaf node
            v = self.get_subscripted_var(N, out_subscr)
            if v:
                used.append(v)

        return unresolved, used

    def extract_defs(self, N, subscr):
        defs = []
        if isinstance(N, list):
            # Todo: Handle exact dataflow for targets
            for tgt in N:
                defs += self.extract_defs(tgt, subscr)

        elif isinstance(N, ast.Tuple):
            for elt in N.elts:
                # Todo: Handle exact dataflow for tuple
                defs += self.extract_defs(elt, subscr)

        else:  # leaf node
            v = self.get_subscripted_var(N, subscr)
            if v:
                defs.append(v)

        return defs

    def replace_unresolved_by_all(self, subscripted_vars, unresolved):
        if not unresolved:
            return subscripted_vars

        updated = []
        for varsub in subscripted_vars:
            sub = tuple(map(lambda d: libtools.all_in_dim if d in unresolved else d, varsub[3]))
            updated.append((varsub[0], varsub[1], varsub[2], sub))
        return updated

    def create_access(self, aname, atype, ast_node, asubscr):
        if isinstance(atype, mytypes.ArrayType):
            indices = []
            bounds = []
            for d in range(atype.ndim):
                if asubscr == libtools.subscr_all or asubscr[d] == libtools.all_in_dim:
                    size = atype.shape[d] if atype.shape else -1
                    if size == 1:
                        idx = 0
                        indices.append(idx)
                    else:
                        v_any = self.new_any_var()
                        indices.append(v_any)
                        lw = 0
                        up = size if isinstance(size, int) and size != -1 or isinstance(size, str) else None
                        b = (lw, v_any, up)
                        bounds.append(b)
                else:
                    indices.append(asubscr[d])

            indices = tuple(indices)
            bounds = tuple(bounds)
        else:
            indices = ()
            bounds = ()

        return StmtInfo.AccessInfo(aname, atype, ast_node, indices, bounds)

    def update_globals_and_depth(self, bounds, accesses, indices_a):
        all = bounds
        for a in accesses:
            if a.bounds:
                all += a.bounds

        for b in all:
            for x in b[0], b[2]:
                if isinstance(x, str):
                    self.globals.add(x)

        d = len(self.outer_bounds) + len(indices_a)
        if d > self.max_depth:
            self.max_depth = d

    def create_perstmt_info(self, N, df, unresolved, used, defed):
        res = []
        for idx in df.indices:
            if idx not in unresolved:
                res.append(idx)
        resolved = tuple(res)
        res_def = self.replace_unresolved_by_all(defed, unresolved)
        res_use = self.replace_unresolved_by_all(used, unresolved)

        if resolved and resolved != libtools.subscr_one:
            assert len(res_def) > 0 and isinstance(res_def[0][1], mytypes.ArrayType)
            indices_a = resolved
            type_d = res_def[0][1]
            sub_d = res_def[0][3]
            shape = type_d.shape if type_d.shape else (-1,) * type_d.ndim
            ups = {x: (shape[sub_d.index(x)] if shape[sub_d.index(x)] != -1 else None) for x in indices_a}
            bounds_a = tuple(map(lambda x: (0, x, ups[x]), indices_a))
        else:
            indices_a = ()
            bounds_a = ()

        if err.verbose_level >= 3:
            err.verbose(3, 'stmt-%d: time_stamp = %s, enclosing bounds = %s' %(self.curr_stmt_id, self.time_stamp, self.outer_bounds))
            err.verbose(3, ' - array indices:', indices_a, 'with bounds:', bounds_a)
            err.verbose(3, ' - use variables:', [(x[0], str(x[1]), x[3]) for x in res_use])
            err.verbose(3, ' - def variables:', [(x[0], str(x[1]), x[3]) for x in res_def])

        bounds_l = tuple(self.outer_bounds)
        indices_l = tuple(map(lambda b: b[1], bounds_l))
        writes = tuple(map(lambda a: self.create_access(a[0], a[1], a[2], a[3]), res_def))
        reads = tuple(map(lambda a: self.create_access(a[0], a[1], a[2], a[3]), res_use))
        schedule = tuple(self.time_stamp)
        for idx in indices_a:
            schedule += (idx, 0)

        s = StmtInfo(self.curr_stmt_id, N, tuple(self.outer_loops),
                     indices_l, indices_a, bounds_l, bounds_a, writes, reads, schedule)
        self.stmts_info.append(s)
        self.update_globals_and_depth(bounds_l + bounds_a, writes + reads, indices_a)

        if err.verbose_level >= 4:
            err.verbose(4, ' -- domain:', s.str_domain())
            err.verbose(4, ' -- writes:', s.str_accesses())
            err.verbose(4, ' -- reads:', s.str_accesses(False))
            err.verbose(4, ' -- schedule:', s.str_schedule())


### AST visitors ###

    def visit_FunctionDef(self, N):
        if not self.is_scop_candidate_func(N):
            return

        err.verbose(2, 'SCoP extract function:', N.name)
        self.init_status()

        # Todo: Need to handle func params as defined at the beginning (liveness for local arrays)
        for stmt in N.body:
            self.visit(stmt)

        self.set_current_scop(N)

    def visit_For(self, N):
        self.update_status_loop_start(N)
        for stmt in N.body:
            self.visit(stmt)
        assert not N.orelse
        self.update_status_loop_end()

    def visit_Assign(self, N):
        df = N.value.func_df if hasattr(N.value, 'func_df') else libtools.default_dfinfo
        err.verbose(3, 'Assign with indices:', df.indices)

        unresolved, used = self.extract_uses(N.value, df.out_subscr)
        # WORKING: dead-code elimination
        defed = self.extract_defs(N.targets, df.out_subscr) if not isinstance(N.targets[0], ast.Name) or N.targets[0].id != 'beamforming' or not isinstance(N.value, ast.Call) or not isinstance(N.value.func, ast.Attribute) or N.value.func.attr != 'zeros' else []
        self.create_perstmt_info(N, df, unresolved, used, defed)
        self.update_status_next_stmt()

    def visit_AugAssign(self, N):
        df = N.func_df if hasattr(N, 'func_df') else libtools.default_dfinfo
        err.verbose(3, 'AugAssign with indices:', df.indices)

        unresolved, used = self.extract_uses(N, df.out_subscr)
        defed = self.extract_defs(N.target, df.out_subscr)
        self.create_perstmt_info(N, df, unresolved, used, defed)
        self.update_status_next_stmt()

    def visit_AnnAssign(self, N):
        if not N.value:
            err.verbose(3, 'Annotation only, skip:', unparse(N))

        df = N.value.func_df if hasattr(N.value, 'func_df') else libtools.default_dfinfo
        err.verbose(3, 'AnnAssign with indices:', df.indices)

        unresolved, used = self.extract_uses(N.value, df.out_subscr)
        defed = self.extract_defs(N.target, df.out_subscr)
        self.create_perstmt_info(N, df, unresolved, used, defed)
        self.update_status_next_stmt()

    def visit_Return(self, N):
        df = N.value.func_df if hasattr(N.value, 'func_df') else libtools.default_dfinfo
        err.verbose(3, 'Return with indices:', df.indices)

        unresolved, used = self.extract_uses(N.value, df.out_subscr)
        # Todo: Modify N such that: tmp = "expression of N.value"; return tmp
        for i in df.out_subscr:
            if i in libtools.index_set:
                unresolved.add(i)
        defed = []
        self.create_perstmt_info(N, df, unresolved, used, defed)
        self.update_status_next_stmt()

class PforInfo:
    def __init__(self, index, start, end, step, output, input, intypes):
        self.index = index
        self.start = start
        self.end = end
        self.step = step
        self.output = output
        self.input = input
        self.intypes = intypes

    def __str__(self):
        range = '%s:%s:%s' %(self.start, self.end, self.step)
        outs = ''
        for a in self.output:
            s = '%s: %s' %(a, self.output[a])
            outs += ', ' + s if outs else s
        if len(self.output) > 1:
            outs = '(' + outs + ')'
        ins = str_iterable(self.input)
        if len(self.input) > 1:
            ins = '(' + ins + ')'
        return 'pfor(index = %s, range = %s, out = %s, in = %s)' %(self.index, range, outs, ins)

class Replacer(ast.NodeTransformer):
    def __init__(self, varmap, accesses = ()):
        for var in varmap:
            assert isinstance(var, str)
        self.varmap = varmap
        self.accesses = accesses

    def visit_Subscript(self, N):
        N.value = self.visit(N.value)
        a = next((a for a in self.accesses if a.ast_node == N), None)
        if a:
            if a.indices_new == (':',) and isinstance(N.value, ast.Name):
                target = N.value
                target.ctx = N.ctx
                N = target
            else:
                # Todo: support
                print(' - accesses:', a.array, ast.dump(N), a.indices, '->', a.indices_new)
                err.error('WIP at midend/polyhedral.py')
        else:
            N.slice = self.visit(N.slice)
        return N

    def visit_Name(self, N):
        if N.id in self.varmap:
            N.id = self.varmap[N.id]
        else:
            a = next((a for a in self.accesses if a.ast_node == N), None)
            if a and a.indices_new != (':',):
                def convert_to_subscript():
                    ctx = N.ctx
                    N.ctx = ast.Load()
                    dims = [ast.Index(ast.Name(str(i), ast.Load())) for i in a.indices_new]
                    return ast.Subscript(value = N, slice = ast.ExtSlice(dims), ctx = ctx)
                N = convert_to_subscript()
        return N

class Scop:
    def __init__(self, stmts_info, max_depth, globals, anys, params,
                 Sdoms, Adoms, space_to_id, space_to_array, W, R, WR, O):
        self.stmts_info = stmts_info
        self.max_depth = max_depth
        self.coef_set = tuple(map(lambda n: '__c_' + str(n), range(max_depth*2+1)))
        self.globals = globals
        self.anys = anys
        self.params = params
        self.Sdoms = Sdoms
        self.Adoms = Adoms
        self.space_to_id = space_to_id
        self.space_to_array = space_to_array
        self.W = W
        self.R = R
        self.WR = WR
        self.O = O
        self.F = None
        self.a2fs = {}
        self.flows = ()
        self.O2 = None

    def space_to_array_search(self, space):
        if space in self.space_to_array:
            return self.space_to_array[space]
        for sp in self.space_to_array:
            if sp == space:
                return self.space_to_array[sp]
        return None

    def dependence_analysis(self):
        revW = self.W.reverse()
        proc = self.O.lex_gt_union_map(self.O)

        if err.verbose_level >= 3:
            str_params = str_iterable(self.params)
        read_list = self.R.get_map_list()
        reads = tuple(map(lambda i: read_list.get_at(i), range(len(read_list))))
        for r in reads:
            acc_r = r.apply_range(revW)
            if acc_r.is_empty():
                continue
            dep_r = acc_r.intersect(proc)
            birth_r = dep_r.apply_range(self.O)
            lastbirth_r = birth_r.lexmax()
            lastprod_r = lastbirth_r.apply_range(self.O.reverse())
            flow_r = lastprod_r.reverse()
            array = self.space_to_array_search(r.range().get_space())
            assert array
            if array in self.a2fs:
                self.a2fs[array] = self.a2fs[array].union(flow_r)
            else:
                self.a2fs[array] = flow_r
            if self.F:
                self.F = self.F.union(flow_r)
            else:
                self.F = flow_r
            if err.verbose_level >= 3:
                err.verbose(3, 'flow (%s): %s' %(array, str(flow_r).replace(str_params, '...')))

        flow_list = self.F.get_map_list()
        self.flows = tuple(map(lambda i: flow_list.get_at(i), range(len(flow_list))))

    def is_fusable(self, base, fusing, level):
        # Todo: skip dependences f that is satisfied outside of 'level'
        ret = -1
        for f in self.flows:
            i = self.space_to_id[f.domain().get_space()]
            j = self.space_to_id[f.range().get_space()]
            if i in base and j in fusing:
                si = self.stmts_info[i]
                sj = self.stmts_info[j]
                assert len(si.schedule_new) > level+1
                if len(sj.schedule_new) <= level+1:
                    return 0

                tmpi = si.schedule_new
                tmpj = sj.schedule_new
                si.schedule_new = [tmpi[level+1]]
                sj.schedule_new = [tmpj[level+1]]
                O_i = isl.Map(si.str_schedule(True))
                O_j = isl.Map(sj.str_schedule(True))
                si.schedule_new = tmpi
                sj.schedule_new = tmpj
                if not f.is_subset(O_i.lex_le_map(O_j)):
                    return 0
                ret = 1
        return ret

    def fusion(self, level, ids):
        err.verbose(3, 'Fusion at level', level, 'for ids:', ids)
        nesting = level//2

        nodes = {}
        for id in ids:
            s = self.stmts_info[id]
            if level == 0:
                s.schedule_new = list(s.schedule_org)
            err.verbose(4, 'id:', id, '/ schedule:', s.schedule_new)
            c0 = s.schedule_new[level]
            c2 = s.schedule_new[level+2] if len(s.schedule_new) > level+2 else 0
            if c0 in nodes:
                nodes[c0].append((id, c2))
            else:
                nodes[c0] = [len(s.indices_l) > nesting, (id, c2)]
        ordered = collections.OrderedDict(sorted(nodes.items()))
        nodes = [ordered[c0] for c0 in ordered]

        nn = len(nodes)
        for i in range(nn):
            n = nodes[i]
            if n[0]:
                for j in range(i+1, nn):
                    m = nodes[j]
                    base = tuple(map(lambda e: e[0], n[1:]))
                    fusing = tuple(map(lambda e: e[0], m[1:]))
                    is_fusable = self.is_fusable(base, fusing, level)
                    if is_fusable == -1:
                        continue  # No dependence between base and fusing

                    idcs = self.stmts_info[fusing[0]].indices
                    if len(idcs) > nesting and is_fusable:
                        err.verbose(3, 'Fused statement(s):', fusing, 'to:', base)
                        lbase = max(tuple(map(lambda e: e[1], n[1:])))
                        mm = [(m[k][0], lbase + k) for k in range(1, len(m))]
                        nodes[j] = n + mm
                        nodes[i] = None
                        err.verbose(4, 'node:', nodes[j])
                    break

        nodes = list(filter(None, nodes))
        for c0 in range(len(nodes)):
            ids2 = []
            for id, c2 in nodes[c0][1:]:
                ids2.append(id)
                s = self.stmts_info[id]
                s.schedule_new[level] = c0
                if len(s.schedule_new) > level+2:
                    s.schedule_new[level+2] = c2
            if len(ids2) >= 2:
                self.fusion(level+2, ids2)

    def is_parallel_loop(self, ids, level):
        O = None
        for id in ids:
            s = self.stmts_info[id]
            if len(s.indices) == 0:
                return False
            tmp = s.schedule_new
            s.schedule_new = [tmp[level]]
            O_i = isl.Map(s.str_schedule(True))
            s.schedule_new = tmp
            if O: O = O.add_map(O_i)
            else: O = isl.UnionMap(O_i)

        lt = O.lex_lt_union_map(O)
        gt = O.lex_gt_union_map(O)
        # Todo: skip dependences from F if they are satisfied outside of 'level'
        return self.F.intersect(lt).is_empty() and self.F.intersect(gt).is_empty()

    def is_parallel_candidate_loop(self, ids, level):
        # WORKING: cost analysis
        return len(ids) >= 1

    def varify_and_set_new_schedules(self):
        O2 = None
        for s in self.stmts_info:
            O2_i = isl.Map(s.str_schedule(True, self.max_depth))
            if O2: O2 = O2.add_map(O2_i)
            else:  O2 = isl.UnionMap(O2_i)

        if self.F.is_subset(O2.lex_lt_union_map(O2)):
            err.verbose(3, 'Checked as valid schedule')
        else:
            err.error('Invalid schedule')

        self.O2 = O2

    def transform(self):
        # self.test1()
        if err.verbose_level >= 3:
            err.verbose(3, 'Initial schedule')
            for s in self.stmts_info:
                err.verbose(3, ' - S%d: %s' %(s.id, s.str_schedule()))

        ids = tuple(map(lambda s: s.id, self.stmts_info))
        self.fusion(0, ids)

        self.varify_and_set_new_schedules()

        if err.verbose_level >= 3:
            err.verbose(3, 'Transformed schedule')
            for s in self.stmts_info:
                err.verbose(3, ' - S%d: %s' %(s.id, s.str_schedule(True)))

    def is_local_array(self, array, filter, lmap):
        if array not in self.a2fs:
            ret = False
            err.verbose(5, 'array:', array, '\n', 'is local:', ret)
            return ret
        flow_a = self.a2fs[array]
        liveness = self.O2.reverse().apply_range(flow_a).apply_range(self.O2)
        filtered = filter.reverse().apply_range(liveness).apply_range(filter)
        ret = not filtered.is_empty() and filtered.is_subset(lmap)
        if err.verbose_level >= 5:
            str_params = str_iterable(self.params)
            err.verbose(5, 'array:', array)
            err.verbose(5, 'flow_a:', str(flow_a).replace(str_params, '...'))
            err.verbose(5, 'liveness:', str(liveness).replace(str_params, '...'))
            err.verbose(5, 'filtered:', str(filtered).replace(str_params, '...'))
            err.verbose(5, 'lmap:', str(lmap).replace(str_params, '...'))
            err.verbose(5, 'is local:', ret)
        return ret

    def update_local_arrays(self, level, id, locals):
        indices = self.coef_set[1:level+1:2]
        coef1_str = str_iterable(self.coef_set)
        coef2_str = str_iterable(self.coef_set[0:level+1])
        filter = isl.Map('{[%s] -> [%s]}' %(coef1_str, coef2_str))
        sche_str = str_iterable(self.stmts_info[id].schedule_new[0:level+1])
        lmap = isl.UnionMap('{[%s] -> [%s]}' %(sche_str, sche_str))

        for a in self.Adoms:
            array = self.space_to_array[a.get_space()]
            if self.is_local_array(array, filter, lmap):
                locals[array] = indices

        err.verbose(4, 'Local arrays:', locals)

    def get_parloop_inout(self, ids, locals, coef):
        output = {}
        input = set()
        intypes = {}

        # Todo: skip dependences f that is satisfied outside of 'level'
        for f in self.flows:
            i = self.space_to_id[f.domain().get_space()]
            j = self.space_to_id[f.range().get_space()]
            if i in ids and j not in ids:
                for w in self.stmts_info[i].writes:
                    if w.array not in locals or coef not in locals[w.array]:
                        err.verbose(4, 'Live-out:', w.array, 'from S%d to S%d' %(i, j))
                        output[w.array] = w.atype
                        assert w.atype.shape and all([s != -1 for s in w.atype.shape])
                        input |= set(s for s in w.atype.shape if isinstance(s, str))

        # Todo: use flows to identify live-in (need to consider func params as write)
        for j in ids:
            for r in self.stmts_info[j].reads:
                if (r.array not in locals or coef not in locals[r.array]) and r.array not in output:
                    err.verbose(4, 'Live-in:', r.array, 'from S? to S%d' %j)
                    input.add(r.array)
                    intypes[r.array] = r.atype

        return output, tuple(input), intypes

    def create_init_statement(self, writes):
        typemap = {}
        for w in writes:
            if not isinstance(w.atype, mytypes.ArrayType):
                err.error('Unsupported type for polyhedral loop fusion:', w.atype)
            if w.array in typemap:
                if typemap[w.array] != w.atype:
                    err.error('Same array with different types is not supported for polyhedral loop fusion')
            else:
                typemap[w.array]= w.atype

        ret = [asttools.init_array_statement(a, typemap[a]) for a in typemap]
        return ret

    def codegen_loop(self, level, ids, locals):
        err.verbose(3, 'codegen_loop at level:', level, 'for ids:', ids)
        depth = (level - 1) // 2
        N = self.stmts_info[ids[0]].outer_loops[depth]
        assert N

        coef = self.coef_set[level]
        self.update_local_arrays(level, ids[0], locals)
        if self.is_parallel_loop(ids, level) and self.is_parallel_candidate_loop(ids, level):
            b = extract_loop_bounds(N)
            output, input, intypes = self.get_parloop_inout(ids, locals, coef)
            par = PforInfo(coef, b[0], b[1], b[2], output, input, intypes)
            err.verbose(3, 'Identified as parallel loop:', par)
        else:
            par = None

        body = self.codegen_body(level+1, ids, locals)
        writes = []
        for id in ids:
            writes += [w for w in self.stmts_info[id].writes if coef in w.indices_new]
        inits = self.create_init_statement(writes)

        N.target = ast.Name(coef, ast.Store())
        N.body = body
        if par:
            N.pfor_info = par
            N.type_comment = str(par)

        if inits: return inits, N
        else: return N

    def contract_function(self, N, contracted):
        # Todo: test impl
        if isinstance(N, ast.Call) and isinstance(N.func, ast.Attribute):
            axis = None
            args = N.args
            keywords = {key.arg: key.value for key in N.keywords}
            if N.func.attr in {'fft', 'ifft'}:
                axis = args[2] if len(args) > 2 else keywords['axis'] if 'axis' in keywords else None
            elif N.func.attr == 'fftshift':
                axis = args[1] if len(args) > 1 else keywords['axes'] if 'axes' in keywords else None
            if axis:
                assert isinstance(axis, ast.Num)
                dim = min(axis.n, len(contracted)-1)
                if dim == axis.n and contracted[dim]:
                    err.error('Illegal transformation with contracted dims', contracted, 'for function call:', unparse(N))
                axis.n -= sum(contracted[:dim+1])

    def update_statement(self, N, idxmap, s, locals):
        raccs = []
        rmap = {}
        for a in s.writes + s.reads:
            ldxs = locals[a.array] if enable_layout_opt and a.array in locals else None
            def replace_index(idx):
                return idxmap[idx] if (idx in idxmap and (ldxs and idxmap[idx] in ldxs or idx in s.indices_a)) else idx
            indices = tuple(map(lambda idx: replace_index(idx), a.indices))
            if indices != a.indices:
                nd1 = len(indices)
                if ldxs:
                    indices = tuple([i for i in indices if i not in locals[a.array]])
                nd2 = len(indices)
                if nd2 < nd1:
                  rmap[a.array] = '%s_%dD' % (a.array, nd2)
                indices = tuple(map(lambda i: ':' if (i in s.indices_a or i in self.anys) else i, indices))
                a.indices_new = indices
                raccs.append(a)
        rmap.update({idx: idxmap[idx] for idx in idxmap if idx in s.indices_l})

        if raccs or rmap:
            err.verbose(4, ' - Replace loop indices:', rmap)
            err.verbose(4, ' - Replace array indices:', ['%s %s -> %s' %(a.array, a.indices, a.indices_new) for a in raccs])
            Replacer(rmap, raccs).visit(N)

        contracted = tuple(map(lambda i: i in idxmap, s.indices_a))
        if any(contracted):
            self.contract_function(N.value, contracted)

    def codegen_stmt(self, level, ids, locals):
        err.verbose(3, 'codegen_stmt at level:', level, 'for ids:', ids)
        id = ids[0]
        s = self.stmts_info[id]
        N = s.ast_node

        coefs = self.coef_set[0:level]
        if coefs:
            sche = s.schedule_new
            idxmap = {sche[i]: coefs[i] for i in range(1, level, 2)}
            self.update_statement(N, idxmap, s, locals)

        return N

    def codegen_body(self, level, ids, locals):
        err.verbose(3, 'codegen_body at level:', level, 'for ids:', ids)
        c_to_ids = {}
        for id in ids:
            c = self.stmts_info[id].schedule_new[level]
            if c in c_to_ids:
                c_to_ids[c].append(id)
            else:
                c_to_ids[c] = [id]

        body = []
        depth = level // 2
        ordered = collections.OrderedDict(sorted(c_to_ids.items()))
        for c in ordered:
            ids = ordered[c]
            if len(ids) >= 2 or len(self.stmts_info[ids[0]].indices_l) > depth:
                l = self.codegen_loop(level+1, ids, locals)
                if isinstance(l, tuple):
                    body += l[0]
                    body.append(l[1])
                else:
                    body.append(l)
            else:
                assert len(ids) == 1
                s = self.codegen_stmt(level, ids, locals)
                body.append(s)
        return body

    def codegen(self, N):
        # self.test2()
        ids = tuple(map(lambda s: s.id, self.stmts_info))
        locals = {}
        body = self.codegen_body(0, ids, locals)
        N.body = body

    def test1(self):
        if len(self.stmts_info) < 7:
            return
        print()
        s5 = self.stmts_info[5]
        s6 = self.stmts_info[6]
        s5.schedule_new = [s5.schedule_org[0], s5.schedule_org[1]]
        s6.schedule_new = [s5.schedule_org[0], s6.schedule_org[1]]
        O_5 = isl.Map(s5.str_schedule(True))
        O_6 = isl.Map(s6.str_schedule(True))
        O = isl.UnionMap(O_5)
        O = O.add_map(O_6)
        lt = O.lex_lt_union_map(O)
        gt = O.lex_gt_union_map(O)
        print('O_5:', O_5)
        print('O_6:', O_6)
        for f in self.flows:
            i = self.space_to_id[f.domain().get_space()]
            j = self.space_to_id[f.range().get_space()]
            if i == 5 and j == 6:
                d = O_5.reverse().apply_range(f).apply_range(O_6)
                print('flow:', f)
                print('diff:', d)
                print('deltas:', d.deltas())
                print('lt:', lt)
                print('gt:', gt)
                print('f.intersect(lt).is_empty():', f.intersect(lt).is_empty())
                print('f.intersect(gt).is_empty():', f.intersect(gt).is_empty())
        print()

    def test2(self):
        print()
        for a in self.Adoms:
            print('Array:', self.space_to_array[a.get_space()])
            rWO_a = self.W.reverse().intersect_domain(a).apply_range(self.O2)
            rRO_a = self.R.reverse().intersect_domain(a).apply_range(self.O2)
            print('rWO_a:', rWO_a)
            print('rRO_a:', rRO_a)

            L = isl.Map("{[c0,c1,c2,c3,c4,c5,c6] -> [c0, c1]}")
            t = rWO_a.apply_range(L).reverse().apply_range(rRO_a.apply_range(L))
            print('L:', L)
            print('t = rWO_a.apply_range(L).reverse().apply_range(rRO_a.apply_range(L)):', t)
            # print('t.deltas:', t.deltas())
            # print('t.deltas is empty:', t.deltas().is_empty())

            lp1map = isl.UnionMap("{[1, c1] -> [1, c1]}")
            print('lp1map:', lp1map)
            print('t.is_subset(lp1map):', t.is_subset(lp1map))

            # print('Array\'s W.lexmin:', rWO_a.lexmin())
            # live = rWO_a.reverse().apply_range(rRO_a)
            # print('Array\'s liveness:', live)
            # print('Array\'s lv.lexmin:', live.lexmin())
        print()

    def test3(self):
        nodes = [None] * (self.stmts_info[-1].schedule_org[0]+1)
        for s in self.stmts_info:
            c0 = s.schedule_org[0]
            loc = s.schedule_org[2] if len(s.schedule_org) > 2 else 0
            if nodes[c0]:
                nodes[c0].append(s.id)
            else:
                nodes[c0] = [s.indices_l != (), s.id]

        for n in nodes:
            if n[0]:
                i = nodes.index(n)
                for m in nodes[i+1:]:
                    if self.is_fusable(n[1:], m[1:], 0):
                        err.verbose(3, 'Fused statement(s):', m[1:], 'to:', n[1:])
                        n.extend(m[1:])
                        nodes.remove(m)
                    else:
                        break

        for c0 in range(len(nodes)):
            for c2 in range(len(nodes[c0]) - 1):
                loc = c2 + 1
                id = nodes[c0][loc]
                s = self.stmts_info[id]
                s.schedule_new = list(s.schedule_org)
                s.schedule_new[0] = c0
                if len(s.schedule_new) > 2:
                    s.schedule_new[2] = c2

    def test4(self, a, filter, lmap):
        rWO_a = self.W.reverse().intersect_domain(a).apply_range(self.O2).apply_range(filter)
        rRO_a = self.R.reverse().intersect_domain(a).apply_range(self.O2).apply_range(filter)
        liveness = rWO_a.reverse().apply_range(rRO_a)
        return not liveness.is_empty() and liveness.is_subset(lmap)
