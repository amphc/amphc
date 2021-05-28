import typed_ast.ast3 as ast
import numpy as np
from typed_astunparse import unparse
import util.error as err
import util.asttools as asttools
from typesys.mytypes import *
from typesys.tools import *
import libbase.entry as libkb

presume_const_annot = False

### Type rules for operators ###
arithmetic1 = {ast.UAdd, ast.USub}
arithmetic2 = {ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv, ast.MatMult}
def is_arithmetic_type(ty):
    return isinstance(ty, NumType) or isinstance(ty, BoolType) or isinstance(ty, AnyType)

bitwise1 = {ast.Invert}
bitwise2 = {ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift}
def is_bitwise_type(ty):
    return isinstance(ty, IntType) or isinstance(ty, BoolType) or isinstance(ty, AnyType)

logical1 = {ast.Not}
logical2 = {ast.And, ast.Or}
def is_logical_type(ty):
    return isinstance(ty, BoolType) or isinstance(ty, AnyType)

comparison = {ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE}
def is_comparison_type(ty):
    return isinstance(ty, NumType) or isinstance(ty, BoolType) or isinstance(ty, StrType) or isinstance(ty, AnyType)

identity = {ast.Is, ast.IsNot}
def is_identity_type(ty):
    return True

membership = {ast.In, ast.NotIn}
def is_membership_type(ty):
    return isinstance(ty, IterableType) or isinstance(ty, CollectiveType) or isinstance(ty, AnyType)


def operator(ty1, ty2, op):
    err.verbose(4, 'rules.operator:', ast.dump(op), 'with types:', ty1, 'and', ty2)

    op_t = type(op)
    if op_t in ty1.operators:
        return call(ty1.operators[op_t], [ty1, ty2])
    elif op_t in ty2.operators:
        return call(ty2.operators[op_t], [ty1, ty2])

    ret = None
    if op_t in arithmetic2:
        if is_arithmetic_type(ty1) and is_arithmetic_type(ty2):
            if op_t is ast.Div:
                ret = type_div(ty1, ty2)
            else:
                ret = type_max(ty1, ty2)

    elif op_t in bitwise2:
        if is_bitwise_type(ty1) and is_bitwise_type(ty2):
            ret = type_max(ty1, ty2)

    if not ret:
        err.error('Unsupported types:', ty1, 'and', ty2, 'for binary operation:', ast.dump(op))

    err.verbose(4, 'Identified as:', ret)
    return ret

def unaryop(ty, op):
    err.verbose(4, 'rules.unaryop:', ast.dump(op), 'with type:', ty)

    op_t = type(op)
    if op_t in ty.operators:
        return call(ty.operators[op_t], [ty])

    ret = None
    if op_t in logical1:
        if is_logical_type(ty):
            ret = BoolType()

    elif op_t in bitwise1:
        if is_bitwise_type(ty):
            ret = ty

    elif op_t in arithmetic1:
        if is_arithmetic_type(ty):
            ret = ty

    if not ret:
        err.error('Unsupported type:', ty, 'for unary operation:', ast.dump(op))

    err.verbose(4, 'Identified as:', ret)
    return ret

def boolop(types, op):
    err.verbose(4, 'rules.boolop:', ast.dump(op), 'with types:', [str(ty) for ty in types])

    op_t = type(op)
    if op_t not in logical2:
        err.error('Unsupported bool operation:', ast.dump(op))

    ret = types[0]
    if op_t not in ret.operators and not is_logical_type(ret):
        err.error('Unsupported type:', ret, 'for bool operation:', ast.dump(op))

    for ty in types[1:]:
        if op_t in ret.operators:
            ret = call(ret.operators[op_t], [ret, ty])
        elif op_t in ty.operators:
            ret = call(ty.operators[op_t], [ret, ty])
        elif not is_logical_type(ty):
            err.error('Unsupported type:', ty, 'for bool operation:', ast.dump(op))

    err.verbose(4, 'Identified as:', ret)
    return ret

def cmpop(types, ops):
    err.verbose(4, 'rules.cmpop:', ast.dump(op), 'with type:', [str(ty) for ty in types])

    n = len(ops)
    assert len(types) == n + 1

    if n == 1:
        op_t = type(ops[0])
        if op_t in types[0].operators:
            return call(types[0].operators[op_t], [types[0], types[1]])
        elif op_t in types[1].operators:
            return call(types[1].operators[op_t], [types[0], types[1]])

    for i in range(n):
        op = ops[i]
        op_t = type(op)
        ty1 = types[i]
        ty2 = types[i+1]
        if op_t in comparison:
            checked = is_comparison_type(ty1) and is_comparison_type(ty2)
        elif op_t in identity:
            checked = is_identity_type(ty1) and is_identity_type(ty2)
        elif op_t in membership:
            checked = is_membership_type(ty1) and is_membership_type(ty2)
        else:
            checked = False

        if not checked:
            err.error('Unsupported types:', ty1, 'and', ty2, 'for compare operation:', ast.dump(op))

    ret = BoolType()
    err.verbose(4, 'Identified as:', ret)
    return ret

def lib_math(type):
    if type == BoolType():
        return FloatType(32)
    elif type <= FloatType(64):
        return FloatType(64)
    elif type <= NumType():
        return type
    else:
        return AnyType()

def call(func_name, types, ktypes = {}, args = [], keywords = {}, mname = None):
    err.verbose(4, 'rules.call:', mname, func_name, 'with types:', [str(ty) for ty in types])
    info = libkb.get_func_info(func_name, mname)
    if not info:
        err.error('Unsupported library function:', mname, func_name)
    ret_type = info.type_rule(types, ktypes, args, keywords)
    func_df = info.dataflow(ret_type, types, ktypes, args, keywords)
    err.verbose(4, 'Identified as:', ret_type, 'with dataflow:', func_df)
    return ret_type, func_df

def size_by_slice_core(slice, base = -1, unsupported_error = True):
    if not slice:
        return base

    if slice.step:
        if unsupported_error: err.error('Slice expression of dimension size cannot contain step, but found:', unparse(slice))
        else: return -1

    size = -1
    if slice.upper:
        if not slice.lower or isinstance(slice.lower, ast.Num) and slice.lower.n == 0:
            size = asttools.extract_value(slice.upper)
        elif isinstance(slice.upper, ast.Num) and isinstance(slice.lower, ast.Num):
            size = slice.upper.n - slice.lower.n
        else:
            if unsupported_error: err.error('Unsupported Slice expression:', unparse(slice))
            else: return -1

        if isinstance(size, int) and size <= 0:
            if unsupported_error: err.error('Slice expression of dimension size must be positive.')
            else: return -1
    else:
        if not slice.lower:
            size = base
        else:
            if unsupported_error:
                err.error('Unsupported Slice expression:', unparse(slice))
            return -1

    return size

def array_subscript(ty, slice):
    assert isinstance(ty, ArrayType)
    if ty.ndim == -1:
        err.error('Subscript expression requires array rank.')
    dims = slice.dims if isinstance(slice, ast.ExtSlice) else [slice]
    n = len(dims)
    is_slice = list(map(lambda x: isinstance(x, ast.Slice), dims)) + [True] * (ty.ndim - n)
    sidxs = np.where(is_slice)[0]
    ndim = len(sidxs)
    if ndim == 0:
        return ty.etype

    base = ty.shape if ty.shape else (-1,) * ty.ndim
    shape = tuple(map(lambda i: size_by_slice_core(dims[i] if i < n else None, base[i], False), sidxs))
    module = ty.module
    return array_with_check(ty.etype, ndim, shape, module)

def subscript(ty, slice):
    err.verbose(4, 'rules.subscript:', ty, 'with:', ast.dump(slice))
    ret = None
    if isinstance(ty, ArrayType):
        ret = array_subscript(ty, slice)
    else:
        # Todo: Handle SparseMat, UTuple, UList, UDict, any, tuple, list, dict
        err.error('WIP in typesys/rules.py: subscript')

    err.verbose(4, 'Identified as:', ret)
    return ret

def iterator(ty):
    return ty.etype if isinstance(ty, IterableType) else AnyType()


### Type annotations ###
def type_by_name(annot):
    assert isinstance(annot, ast.Name)

    name = annot.id
    if name is 'int32' or name is 'int':
        return IntType(32)
    elif name is 'int64' or name is 'long':
        return IntType(64)
    elif name is 'float32' or name is 'float':
        return FloatType(32)
    elif name is 'float64' or name is 'double':
        return FloatType(64)
    elif name is 'complex64':
        return ComplexType(64)
    elif name is 'complex128':
        return ComplexType(128)
    elif name is 'num':
        return NumType()
    elif name is 'bool':
        return BoolType()
    elif name is 'str':
        return StrType()
    elif name is 'list':
        return ListType()
    elif name is 'tuple':
        return TupleType()
    elif name is 'set':
        return SetType()
    elif name is 'dict':
        return DictType()
    elif name is 'any' or name is 'Any':
        return AnyType()
    else:
        err.error('Unsupported annotation type:', name)

    assert False  # unreachable here

def type_by_name_const(annot):
    assert isinstance(annot, ast.NameConstant)
    return name_constant(annot.value)

def type_by_attribute(annot):
    assert isinstance(annot, ast.Attribute)
    if not isinstance(annot.value, ast.Name):
        err.error('Unsupported annotation format:', unparse(annot))

    attr = annot.attr
    mvar = annot.value.id
    minfo = libkb.var_to_module(mvar, True)
    if not minfo:
        err.error('Unsupported or not-imported module:', mvar, 'for attribute annotation')

    if hasattr(minfo, 'dtypes') and attr in minfo.dtypes:
        return minfo.dtypes[attr], minfo.mname
    elif hasattr(minfo, 'atypes') and attr in minfo.atypes:
        return minfo.atypes[attr], minfo.mname
    else:
        err.error('Unsupported annotation format:', unparse(annot))

def type_by_call(annot):
    assert isinstance(annot, ast.Call)
    if not isinstance(annot.func, ast.Name):
        err.error('Unsupported annotation format:', unparse(annot))

    args = annot.args
    nargs = len(args)
    etype = annotation(args[0], False)

    name = annot.func.id
    if name in {'UTuple', 'UList', 'USet', 'SparseMat'} and nargs != 1:
            err.error(name, 'type needs one argument for element type annotation.')
    elif name is 'UTuple':
        return UTupleType(etype)
    elif name is 'UList':
        return UListType(etype)
    elif name is 'USet':
        return USetType(etype)
    elif name is 'SparseMat':
        return SparseMatType(etype)
    elif name is 'Array':
        if nargs != 2 or not isinstance(args[1], ast.Num):
            err.error('Array type needs two arguments for element type annotation & # dimensions.')
        ndim = int(args[1].n)
        return ArrayType(etype, ndim)  # Todo: Support cupy.Array to extract module from annot
    elif name is 'UDict':
        if nargs != 2:
            err.error('UDict type needs two arguments for key & type annotations.')
        vtype = annotation(args[1], False)
        return UDictType(etype, vtype)
    else:
        err.error('Unsupported annotation type:', name)

    assert False  # unreachable here

def size_by_slice(slice):
    size = -1
    if isinstance(slice, ast.Index):
        size = asttools.extract_value(slice.value)
        if isinstance(size, int) and size <= 0:
            err.error('Slice expression of dimension size must be positive.')
    elif isinstance(slice, ast.Slice):
        size = size_by_slice_core(slice)
    else:
        err.error('Unsupported Slice expression:', unparse(slice))

    return size

def type_by_subscript(annot):
    assert isinstance(annot, ast.Subscript)

    v = annot.value
    if isinstance(v, ast.Name):
        etype = type_by_name(v)
        module = ''
    elif isinstance(v, ast.Attribute):
        etype, module = type_by_attribute(v)
        if isinstance(etype, ArrayType):
            err.error('Unsupported annotation format:', unparse(annot))
    elif (isinstance(v, ast.Subscript) and isinstance(v.value, ast.Attribute)
          and isinstance(v.slice, ast.Index) and isinstance(v.slice.value, ast.Attribute)):
        tmp, module = type_by_attribute(v.value)
        etype, tmp = type_by_attribute(v.slice.value)
    else:
        err.error('Unsupported annotation format:', unparse(annot))

    slice = annot.slice
    if isinstance(slice, ast.ExtSlice):
        shape = tuple(map(size_by_slice, slice.dims))
    elif isinstance(slice, ast.Index) and isinstance(slice.value, ast.Tuple):
        shape = tuple(map(asttools.extract_value, slice.value.elts))
    else:
        shape = (size_by_slice(slice),)  # One element tuple

    global presume_const_annot
    if not presume_const_annot:
        shape = tuple(map(lambda s: -1 if isinstance(s, int) and s > 1 else s, shape))
    ndim = len(shape)
    return array_with_check(etype, ndim, shape, module)

def annotation(annot, top = True):
    if top:
        err.verbose(4, 'rules.annotation:', (ast.dump(annot) if annot else None))

    if not annot:
        ret = AnyType()
    elif isinstance(annot, ast.Name):
        ret = type_by_name(annot)
    elif isinstance(annot, ast.NameConstant):
        ret = type_by_name_const(annot)
    elif isinstance(annot, ast.Attribute):
        ret, tmp = type_by_attribute(annot)
    elif isinstance(annot, ast.Call):
        ret = type_by_call(annot)
    elif isinstance(annot, ast.Subscript):
        ret = type_by_subscript(annot)
    elif isinstance(annot, ast.Tuple):
        ret = tuple(map(lambda x: annotation(x, False), annot.elts))
    else:
        err.error('Unsupported annotation:', unparse(annot))

    if top:
        if isinstance(ret, tuple): err.verbose(4, 'Identified as:', [str(x) for x in ret])
        else: err.verbose(4, 'Identified as:', ret)
        
    return ret


### Type rules for constant ###
def num(n):
    if isinstance(n, int):
        return IntType(32)
    elif isinstance(n, float):
        return FloatType(32)
    elif isinstance(n, complex):
        return ComplexType(64)
    else:
        err.error('Unsupported constant number:', n, '; with type:', type(n))

def name_constant(value):
    if isinstance(value, bool):
        return BoolType()
    elif isinstance(value, int):
        return IntType(32)
    elif isinstance(value, float):
        return FloatType(32)
    elif isinstance(value, complex):
        return ComplexType(64)
    elif not value:
        return AnyType()
    else:
        err.error('Unsupported constant name:', value, '; with type:', type(value))

def string():
    return StrType()
