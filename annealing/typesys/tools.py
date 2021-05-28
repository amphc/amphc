import typed_ast.ast3 as ast
import util.error as err
import util.asttools as asttools
from typesys.mytypes import *

### Tools to manage type conversion ###
def width_max(ty1, ty2):
    wd1 = ty1.width if isinstance(ty1, NumType) else -1
    if isinstance(ty1, ComplexType):
        wd1 = int(wd1 / 2)  # ComplexType.width is total for real and imaginary

    wd2 = ty2.width if isinstance(ty2, NumType) else -1
    if isinstance(ty2, ComplexType):
        wd2 = int(wd1 / 2)  # ComplexType.width is total for real and imaginary

    return max(wd1, wd2)

def type_max(ty1, ty2):
    # MyType's magic methods (e.g., __lt__) are used
    ty = max(ty1, ty2)

    if isinstance(ty, BoolType) or isinstance(ty, AnyType) or type(ty) is NumType:
        return ty

    wd = width_max(ty1, ty2)
    if isinstance(ty, ComplexType):
        return ComplexType(wd * 2)  # ComplexType.width is total for real and imaginary
    elif isinstance(ty, NumType):
        return ty.__class__(wd)
    else:
        err.error('Unsupported type_max on types:', ty1, ty2)

def type_div(ty1, ty2):
    # MyType's magic methods (e.g., __lt__) are used
    ty = max(ty1, ty2)

    if isinstance(ty, BoolType):
        return FloatType(32)

    wd = width_max(ty1, ty2)
    bound = FloatType(wd)
    if ty <= bound:
        return bound

    bound = ComplexType(wd * 2)
    if ty <= bound:
        return bound

    if isinstance(ty, NumType) or isinstance(ty, AnyType):
        return ty
    else:
        err.error('Unsupported type_div on types:', ty1, ty2)

def type_merge(ty1, ty2):
    # MyType's magic method __eq__ is used
    if ty1 == ty2:
        return ty1

    elif isinstance(ty1, NumType) and isinstance(ty2, NumType):
        return NumType()

    elif isinstance(ty1, IterableType) and type(ty1) is type(ty2):
        cl = type(ty1)
        etype = type_merge(ty1.etype, ty2.etype)
        if cl is ArrayType:
            # Handling array -- Todo: merge shape info;  Q.: to be error when ndim = -1?
            ndim = ty1.ndim if ty1.ndim == ty2.ndim else -1
            if ty1.module == ty2.module:
                module = ty1.module
                return ArrayType(etype, ndim, module = module)
            else:
                return AnyType()

        elif cl is UDictType:
            return cl(etype, type_merge(ty1.vtype, ty2.vtype))
        else:
            return cl(etype)

    elif isinstance(ty1, CollectiveType) or isinstance(ty2, CollectiveType):
        cl1 = type(ty1)
        cl2 = type(ty2)
        if (cl1 is TupleType and cl2 is UTupleType) or (cl2 is TupleType and cl1 is UTupleType):
            return TupleType()
        elif (cl1 is ListType and cl2 is UListType) or (cl2 is ListType and cl1 is UListType):
            return ListType()
        elif (cl1 is SetType and cl2 is USetType) or (cl2 is SetType and cl1 is USetType):
            return SetType()
        elif (cl1 is DictType and cl2 is UDictType) or (cl2 is DictType and cl1 is UDictType):
            return DictType()

    return AnyType()

def type_including(ty1, ty2):
    return ty1 == type_merge(ty1, ty2)


### Tools related to MyType allocation ###

def array_with_check(etype, ndim, shape, module = ''):
    assert not shape or ndim == len(shape)
    if not shape or min(tuple(map(lambda x: x == -1, shape))):
        # empty shape or all elements of shape is -1
        if module: return ArrayType(etype, ndim, module = module)
        else: return ArrayType(etype, ndim)
    else:
        if module: return ArrayType(etype, ndim, shape, module)
        else: return ArrayType(etype, ndim, shape)


### User function signatures ###
def convert_tuple_type(tupled_type):
    if len(tupled_type) == 1 or all([tupled_type[0] == ty for ty in tupled_type[1:]]):
        return UTupleType(tupled_type[0])
    else:
        return TupleType()

class TypedSignature:
    def __init__(self, name, params, returns):
        ids = set(map(lambda x: x[0], params))
        if len(params) != len(ids):
            err.error('Duplicated parameters in:', params, 'for function:', name)

        self.name = name
        self.params = params    # list of tuples (name, type, default)
        self.returns = returns  # type

    def verify_args(self, types, keys = {}):
        nargs = len(types)
        nparams = len(self.params)
        if nargs > nparams:
            err.error('Calling:', self.name, 'with', nargs, 'arguments while it has only', nparams, 'parameters.')

        for i in range(nargs):
            ty = convert_tuple_type(types[i]) if isinstance(types[i], tuple) else types[i]
            if not isinstance(ty, AnyType) and not type_including(self.params[i][1], ty):
                err.warning('Calling:', self.name, 'with unmatched argument type:', ty, 'at position:', i)

        for i in range(nargs, nparams):
            n = self.params[i][0]
            if n in keys:
                ty = convert_tuple_type(keys[n]) if isinstance(keys[n], tuple) else keys[n]
                if not isinstance(ty, AnyType) and not type_including(self.params[i][1], ty):
                    err.warning('Calling:', self.name, 'with unmatched argument type:', ty, 'for parameter:', n)
            elif not self.params[i][2]:
                err.error('Calling:', self.name, 'without argument for parameter:', n)

        err.verbose(2, 'Verified argument types for:', self.name)

    def params_as_dict(self):
        return { p[0]: p[1] for p in self.params }


### Handling flow-sensitive typing rules for control branches ###
class BranchedContainer:
    def __init__(self):
        self.parent = None
        self.body = None
        self.orelse = None

    def fork(self):
        self.body = self.__class__()
        self.orelse = self.__class__()
        self.body.parent = self
        self.orelse.parent = self

    def join(self):
        self.body.parent = None
        self.orelse.parent = None
        self.body = None
        self.orelse = None

class TypeTable(BranchedContainer):
    def __init__(self, init = {}):
        self.type_table = init
        BranchedContainer.__init__(self)

    def add(self, var, ty):
        if not self.type_table:
            self.type_table = {var: ty}
        elif var in self.type_table:
            if isinstance(self.type_table[var], list):
                self.type_table[var].append(ty)
            else:
                self.type_table[var] = [self.type_table[var], ty]
        else:
            self.type_table[var] = ty

    def get_local(self, var, ver = -1):
        if var not in self.type_table:
            return None

        types = self.type_table[var]
        if isinstance(types, list):
            if ver > len(types):
                err.error('Exceeded version number:', ver, '; must be <', len(types))
            return types[ver]
        else:
            if ver != -1 and ver != 0:
                err.error('Exceeded version number:', ver, '; must be 0 or -1')
            return types

    def get(self, var, ver = -1):
        if var not in self.type_table:
            if self.parent:
                return self.parent.get(var, ver)
            return None

        return self.get_local(var, ver)

    def join(self):
        bkeys = self.body.type_table.keys()
        ekeys = self.orelse.type_table.keys()
        cand = bkeys & ekeys
        for var in bkeys | ekeys:
            ty = AnyType()
            if var in cand:
                bty = self.body.get_local(var)
                ety = self.orelse.get_local(var)
                ty = type_merge(bty, ety)
            self.add(var, ty)
        BranchedContainer.join(self)

    def __str__(self):
        ret = 'type table: { '
        for var in self.type_table:
            ret += str(var) + ': '
            types = self.type_table[var]
            if isinstance(types, MyType):
                ret += str(types) + ', '
            else:
                ret += str([str(ty) for ty in types]) + ', '
        ret += '}'

        if self.body:
            ret += ' / body ' + str(self.body)
        if self.orelse:
            ret += ' / orelse ' + str(self.orelse)

        return ret
