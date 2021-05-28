import typed_ast.ast3 as ast
import util.error as err
import util.asttools as asttools
import typesys.mytypes as mytypes
import typesys.tools as tytools
from libbase.tools import *


class FuncInfo:
    def __init__(self, pureness = False):
        self.pureness = pureness

    def type_rule(self, arg_types, key_types, args, keywords):
        err.error('Need to be implement type_rule in libbase/libfuncs.py')

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        err.error('Need to be implement dataflow in libbase/libfuncs.py')

    def computation(self):
        err.error('Need to be implement computation in libbase/libfuncs.py')

class PureBaseInfo(FuncInfo):
    def __init__(self):
        FuncInfo.__init__(self, True)

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        return default_dfinfo  # Todo: more accurate?

class PureArrayBaseInfo(PureBaseInfo):
    def __init__(self, module = ''):
        PureBaseInfo.__init__(self)
        self.module = module

    def check_module(self, ty):
        if self.module and isinstance(ty, mytypes.ArrayType) and self.module != ty.module:
            err.error('Inconsistent modules: function module', self.module, 'vs. array\'s module', ty.module)


class IntInfo(PureBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        return mytypes.IntType(32)

class RangeInfo(PureBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        assert arg_types
        return mytypes.IterableType(arg_types[0])

class ArangeInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        d = args[3] if len(args) > 3 else keywords['dtype'] if 'dtype' in keywords else None
        if d:
            etype = etype_from_ast(d)
        else:
            aty0 = arg_types[0] if len(arg_types) > 1 else key_types['start'] if 'start' in key_types else None
            aty1 = arg_types[1] if len(arg_types) > 1 else arg_types[0] if len(arg_types) == 1 else keywords['stop'] if 'stop' in keywords else None
            aty2 = arg_types[2] if len(arg_types) > 2 else key_types['step'] if 'step' in key_types else None
            assert aty1
            etype = aty1
            for t in aty0, aty2, mytypes.IntType(64):
                if t:
                    etype = tytools.type_max(t, etype)
        return tytools.array_with_check(etype, 1, (), module = self.module)

class LinSpaceInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        # WORKING: support other args, e.g., restep
        aty0 = arg_types[0] if len(arg_types) > 0 else key_types['start'] if 'start' in key_types else None
        aty1 = arg_types[1] if len(arg_types) > 1 else key_types['stop'] if 'stop' in key_types else None
        assert aty0 and aty1
        etype = tytools.type_max(aty0, aty1)
        return tytools.array_with_check(etype, 1, (), module = self.module)

class NewArrayInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        s = args[0] if len(args) > 0 else keywords['shape'] if 'shape' in keywords else None
        d = args[1] if len(args) > 1 else keywords['dtype'] if 'dtype' in keywords else None
        assert s
        shape = shape_from_ast(s)
        etype = etype_from_ast(d)
        return tytools.array_with_check(etype, len(shape), shape, self.module)

class RandnInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        if keywords:
            err.error('Keyword arguments are currently unsupported for randn.')
        etype = mytypes.FloatType(64)
        if not args:
            return etype
        else:
            shape = shape_from_ast(args)
            return tytools.array_with_check(etype, len(shape), shape, self.module)

class CopyInfo(PureArrayBaseInfo):
    def type_rule_base(self, arg_types, key_types, args, keywords, to_check = True):
        aty = arg_types[0] if len(arg_types) > 0 else key_types['a'] if 'a' in key_types else None
        assert aty
        if to_check:
            self.check_module(aty)  # Illegal to copy array across devices (numpy vs. cupy)
        return aty

    def type_rule(self, arg_types, key_types, args, keywords):
        return self.type_rule_base(arg_types, key_types, args, keywords)

class CopyToHostInfo(CopyInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        aty = self.type_rule_base(arg_types, key_types, args, keywords, False)
        assert isinstance(aty, mytypes.ArrayType)
        return tytools.array_with_check(aty.etype, aty.ndim, aty.shape, 'numpy')

class AsArrayInfo(CopyInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        aty = self.type_rule_base(arg_types, key_types, args, keywords, False)
        if isinstance(aty, mytypes.ArrayType):
            assert isinstance(aty, mytypes.ArrayType)
            return tytools.array_with_check(aty.etype, aty.ndim, aty.shape, self.module)
        else:
            etype = aty.etype if isinstance(aty, mytypes.IterableType) else mytypes.NumType()
            return tytools.array_with_check(etype, 1, (), self.module)

class RavelInfo(CopyInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        aty = self.type_rule_base(arg_types, key_types, args, keywords)
        assert isinstance(aty, mytypes.ArrayType)
        return tytools.array_with_check(aty.etype, 1, (), module = aty.module)

class SqueezeInfo(CopyInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        aty = self.type_rule_base(arg_types, key_types, args, keywords)
        assert isinstance(aty, mytypes.ArrayType)
        if aty.shape:
            shape = squeeze(aty.shape)
            return tytools.array_with_check(aty.etype, len(shape), shape, aty.module)
        else:
            return mytypes.ArrayType(aty.etype, -1, module = aty.module)

class ReshapeInfo(PureArrayBaseInfo):
    def type_rule_base(self, arg_types, key_types, args, keywords):
        aty = arg_types[0] if len(arg_types) > 0 else key_types['a'] if 'a' in key_types else None
        s = args[1] if len(args) > 1 else keywords['shape'] if 'shape' in keywords else None
        assert aty and isinstance(aty, mytypes.ArrayType) and s
        self.check_module(aty)
        shape = shape_from_ast(s)
        return shape, aty

    def type_rule(self, arg_types, key_types, args, keywords):
        shape, aty = self.type_rule_base(arg_types, key_types, args, keywords)
        return tytools.array_with_check(aty.etype, len(shape), shape, aty.module)

class TileInfo(ReshapeInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        tile, aty = self.type_rule_base(arg_types, key_types, args, keywords)
        if aty.ndim != -1:
            ndim, shape = shape_tile(aty.ndim, aty.shape, tile)
            return tytools.array_with_check(aty.etype, ndim, shape, aty.module)
        else:
            return mytypes.ArrayType(aty.etype, -1, module = aty.module)


class TransposeInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        aty = arg_types[0]
        assert aty and isinstance(aty, mytypes.ArrayType)
        if aty.ndim == 1 or aty.ndim == -1:
            return aty
        else:
            etype = aty.etype
            shape = transpose(aty.shape, aty.ndim)
            module = aty.module
            return tytools.array_with_check(etype, len(shape), shape, module)

    # Todo: Define dataflow

class UnaryInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        import typesys.rules as rules
        aty = arg_types[0] if len(arg_types) > 0 else key_types['x'] if 'x' in key_types else None
        assert aty
        self.check_module(aty)
        (etype0, ndim, shape) = array_type_extract(aty)
        etype = rules.lib_math(etype0) # Todo: confirm this type rule
        if ndim == 0:
            return etype
        else:
            module = aty.module
            return tytools.array_with_check(etype, ndim, shape, module)

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        aty = arg_types[0] if len(arg_types) > 0 else key_types['x'] if 'x' in key_types else None
        assert aty and ret_type

        if isinstance(ret_type, mytypes.ArrayType):
            assert ret_type.ndim > 0 and isinstance(aty, mytypes.ArrayType) and aty.ndim > 0
            global index_set
            indices = index_set[0:ret_type.ndim]
            out_subscr = dataflow_broadcast(ret_type, indices)
            in_subscr = dataflow_broadcast(aty, indices)
            return DataFlowInfo(indices, out_subscr, in_subscr)
        else:
            return default_dfinfo  # Todo: more accurate?

class ArithmeticInfo(PureArrayBaseInfo):
    def __init__(self, op, module = ''):
        PureArrayBaseInfo.__init__(self, module)
        self.op = op

    def type_rule(self, arg_types, key_types, args, keywords):
        import typesys.rules as rules
        # WORKING: use key_types if needed
        (etype1, ndim1, shape1) = array_type_extract(arg_types[0])
        (etype2, ndim2, shape2) = array_type_extract(arg_types[1])
        assert ndim1 != 0 or ndim2 != 0

        self.check_module(arg_types[0])
        self.check_module(arg_types[1])
        module = arg_types[0].module if ndim1 != 0 else arg_types[1].module

        if 'dtype' in keywords:
            etype = etype_from_ast(keywords['dtype'])
        else:
            etype = rules.operator(etype1, etype2, self.op)

        if ndim1 >= 0 and ndim2 >= 0:
            ndim, shape = shape_broadcast(ndim1, ndim2, shape1, shape2)
            return tytools.array_with_check(etype, ndim, shape, module)
        else:
            return mytypes.ArrayType(etype, -1, module = module)

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        # WORKING: use key_types if needed
        assert isinstance(ret_type, mytypes.ArrayType) and ret_type.ndim > 0
        global index_set

        indices = index_set[0:ret_type.ndim]
        out_subscr = dataflow_broadcast(ret_type, indices)
        in_subscr = [dataflow_broadcast(ty, indices) for ty in arg_types]
        indices = out_subscr
        return DataFlowInfo(indices, out_subscr, in_subscr)

class MatMultInfo(PureArrayBaseInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        import typesys.rules as rules
        # WORKING: use key_types if needed
        (etype1, ndim1, shape1) = array_type_extract(arg_types[0])
        (etype2, ndim2, shape2) = array_type_extract(arg_types[1])
        assert ndim1 != 0 and ndim2 != 0

        self.check_module(arg_types[0])
        self.check_module(arg_types[1])
        module = arg_types[0].module

        etype = rules.operator(etype1, etype2, ast.Mult())
        if ndim1 == 1 and ndim2 == 1:
            return etype
        elif ndim1 >= 1 and ndim2 >= 1:
            ndim, shape = shape_matmult(ndim1, ndim2, shape1, shape2)
            return tytools.array_with_check(etype, ndim, shape, module)
        else:
            return mytypes.ArrayType(etype, -1, module = module)

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        return default_dfinfo # WORKING: test impl

class FFTInfo(PureArrayBaseInfo):
    def type_rule_base(self, arg_types, key_types, N_size, N_axis):
        aty = arg_types[0] if len(arg_types) > 0 else key_types['a'] if 'a' in key_types else None
        assert aty
        self.check_module(aty)
        module = aty.module
        etype = aty.etype
        ndim = aty.ndim
        axis = asttools.extract_value(N_axis) if N_axis else ndim - 1
        size = asttools.extract_value(N_size) if N_size else -1
        shape = shape_ope1d_on_md(ndim, aty.shape, axis, size)
        return tytools.array_with_check(etype, ndim, shape, module)

    def dataflow_base(self, ret_type, N_axis, nargs):
        global index_set, all_in_dim, subscr_one
        assert isinstance(ret_type, mytypes.ArrayType) and ret_type.ndim > 0

        ndim = ret_type.ndim
        axis = asttools.extract_value(N_axis) if N_axis else ndim - 1

        if ndim == 1:
            indices = subscr_one
            out_subscr = (all_in_dim,)
        else:
            indices = index_set[0:ndim-1]
            out_subscr = dataflow_ope1d_on_md(ndim, axis, indices)

        if nargs == 1:
            in_subscr = out_subscr
        else:
            in_subscr = [out_subscr] + [subscr_one] * (nargs - 1)

        return DataFlowInfo(indices, out_subscr, in_subscr)

    def type_rule(self, arg_types, key_types, args, keywords):
        n = args[1] if len(args) > 1 else keywords['n'] if 'n' in keywords else None
        axis = args[2] if len(args) > 2 else keywords['axis'] if 'axis' in keywords else None
        return self.type_rule_base(arg_types, key_types, n, axis)

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        axis = args[2] if len(args) > 2 else keywords['axis'] if 'axis' in keywords else None
        return self.dataflow_base(ret_type, axis, len(arg_types) + len(key_types))

class FFTShiftInfo(FFTInfo):
    def type_rule(self, arg_types, key_types, args, keywords):
        aty = arg_types[0] if len(arg_types) > 0 else key_types['a'] if 'a' in key_types else None
        assert aty
        return aty

    def dataflow(self, ret_type, arg_types, key_types, args, keywords):
        axes = args[1] if len(args) > 1 else keywords['axes'] if 'axes' in keywords else None
        assert not axes or isinstance(axes, ast.Num) or isinstance(axes, ast.Name) # WORKING: test impl
        return self.dataflow_base(ret_type, axes, len(arg_types) + len(key_types))
