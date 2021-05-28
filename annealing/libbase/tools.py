import typed_ast.ast3 as ast
import util.error as err
import util.asttools as asttools
import typesys.mytypes as mytypes

# Settings for dataflow analysis
max_num_indices = 20
dataflow_use_symbols = True
if dataflow_use_symbols:
    import sympy as sym


### Tools to parse ast ###

def shape_from_ast(N):
    if isinstance(N, ast.Tuple):
        shape0 = tuple(map(asttools.extract_value, N.elts))
    elif isinstance(N, list):
        shape0 = tuple(map(asttools.extract_value, N))
    else:
        shape0 = (asttools.extract_value(N),)
    shape = tuple(map(lambda x: x if isinstance(x, int) or isinstance(x, str) else -1, shape0))
    return shape

def etype_from_ast(N):
    import typesys.rules as rules
    if not N:
        return mytypes.FloatType(64)
    elif isinstance(N, ast.Attribute):
        ty, md = rules.type_by_attribute(N)
        return ty
    else:
        err.error('Unsupported dtype information for array allocation:', unparse(N))


### Tools to infer array shape ###

def array_type_extract(ty):
    return (ty.etype, ty.ndim, ty.shape) if isinstance(ty, mytypes.ArrayType) else (ty, 0, None)

def squeeze(shape):
    squeezed = tuple()
    for s in shape:
        if s != 1:
            squeezed = squeezed + (s,)
    return squeezed

def transpose(shape, ndim):
    if not shape:
        return (-1,) * ndim
    assert len(shape) == ndim
    if ndim == 2:
        return (shape[1], shape[0])
    else:
        err.error('Unsupported rank for transpose:', ndim)

def extend_with(shape, ndim, val):
    for i in range(ndim - len(shape)):
        shape = (val,) + shape
    return shape

def compare_symbolic(val1, val2):
    if isinstance(val1, int):
        return val2
    elif isinstance(val2, int):
        return val1

    # Todo: 1) general way to handle symbols; 2) collect symbols for runtime broadcastable check
    ret = val1 if isinstance(val1, str) else val2
    err.warning('Different symbols are found in shape inference:', val1, 'vs.', val2, '; inferred as:', ret)

    return ret

def shape_tile(ndim1, shape1, tile):
    assert ndim1 > 0 and tile and all([t != -1 for t in tile])

    ndim = max(ndim1, len(tile))
    if not shape1:
        shape1 = (-1,) * ndim1

    shape1 = extend_with(shape1, ndim, 1)
    tile = extend_with(tile, ndim, 1)
    def mul(size, factor):
        if size == -1:
            return -1
        elif isinstance(size, int) and isinstance(factor, int):
            return size * factor
        elif size == 1:
            return factor
        elif factor == 1:
            return size
        else:
            return -1  # Todo: multiply symbol(s)
    shape = tuple(map(lambda v: mul(v[0], v[1]), zip(shape1, tile)))

    return ndim, shape

def shape_broadcast(ndim1, ndim2, shape1, shape2):
    assert (not shape1 or len(shape1) == ndim1) and (not shape2 or len(shape2) == ndim2)

    ndim = max(ndim1, ndim2)
    if not shape1 and not shape2:
        shape = ()
    elif shape1 and shape2:
        shape1 = extend_with(shape1, ndim, 1)
        shape2 = extend_with(shape2, ndim, 1)
        def bc(v1, v2):
            if v1 == v2:
                return v1
            elif isinstance(v1, int) and isinstance(v2, int):
                if v1 >= 2 and v2 >= 2:
                    err.error('Cannot broadcast between:', shape1, 'and', shape2)
                else:
                    v = max(v1,v2)
                    if v == 1 and v1 == -1 or v2 == -1:
                        v = -1
                    return v
            else:
                return compare_symbolic(v1, v2)
        shape = tuple(map(lambda v: bc(v[0], v[1]), zip(shape1, shape2)))
    else:
        tmp = shape1 if shape1 else shape2
        shape = extend_with(tmp, ndim, -1)

    return ndim, shape

def shape_matmult(ndim1, ndim2, shape1, shape2):
    assert (ndim1 >= 1 and ndim2 >= 1 and ndim1 + ndim2 >= 3
            and (not shape1 or len(shape1) == ndim1) and (not shape2 or len(shape2) == ndim2))

    if ndim1 == 1 and ndim2 == 2:
        ndim = 1
        size = shape2[1] if shape2 else -1
        shape = (size,)
    elif ndim1 == 2 and ndim2 == 1:
        ndim = 1
        size = shape1[0] if shape1 else -1
        shape = (size,)
    elif ndim1 == 2 and ndim2 == 2:
        ndim = 2
        size1 = shape1[0] if shape1 else -1
        size2 = shape2[1] if shape2 else -1
        shape = (size1, size2)
    else:
        err.error('Unsupported argument ranks for matmult:', ndim1, ndim2)
    return ndim, shape

def shape_ope1d_on_md(ndim, base, axis, size):
    if not base:
        base =  (-1,) * ndim
    else:
        assert len(base) == ndim

    if size == -1:
        return base
    else:
        return tuple(map(lambda i: size if i == axis else base[i], range(ndim))) 


### Dataflow analysis for library function ###

def replace_vars(collection, mapping):
    if isinstance(collection, tuple):
        return tuple(mapping[e] if e in mapping else e for e in collection)
    elif isinstance(collection, list):
        return [replace_vars(c, mapping) for c in collection]
    else:
        err.error('Unsupported collection:', collection)

'''
Dataflow relation between input array(s) and output array,
where the computation is abstracted as n-dimensional space:
 - indices: coordinate (i_0, i_1, ... i_n-1) of n-D space
 - out_subscr: subscript of output array element that is written by indices
 - in_subscr: subscripts of input input array elements that are read by indices

Todo: handle tupled output and inputs, e.g., out_subscr = ((i0, i1), (i1, i0))
'''
class DataFlowInfo:
    def __init__(self, indices, out_subscr, in_subscr):
        self.indices = indices
        self.out_subscr = out_subscr  # one-to-one mapping with indices, e.g., id(i0,i1) - out(i1,i0)
        self.in_subscr = in_subscr    # list to describe each arg, one entry to describe all args

    def __str__(self):
        ret = 'indices: %s, out: %s, in: %s' % (self.indices, self.out_subscr, self.in_subscr)
        return ret

    def get_assigned(self, assigned_subscr):
        global subscr_all, all_in_dim

        if self.out_subscr == assigned_subscr:
            return self

        m = len(assigned_subscr)
        if self.out_subscr == subscr_all and m > 1:
            out_subscr = (all_in_dim,) * m
            return self.__class__(self.indices, out_subscr, self.in_subscr)

        n = len(self.out_subscr)
        if assigned_subscr == subscr_all:
            assigned_subscr = (all_in_dim,) * n

        assert m == n
        mapping = {self.out_subscr[i]: (assigned_subscr[i] if self.out_subscr[i] in self.indices
                                        else self.out_subscr[i]) for i in range(n)}
        indices = replace_vars(self.indices, mapping)
        out_subscr = replace_vars(self.out_subscr, mapping)
        in_subscr = replace_vars(self.in_subscr, mapping)
        return self.__class__(indices, out_subscr, in_subscr)

if dataflow_use_symbols:
    index_set = tuple(map(lambda n: sym.Symbol('i' + str(n)), range(max_num_indices)))
    all_in_dim = sym.Symbol(':')
    all_elms = sym.Symbol('*')
else:
    index_set = tuple(map(lambda n: 'i' + str(n), range(max_num_indices)))
    all_in_dim = ':'
    all_elms = '*'

subscr_all = (all_elms,)
subscr_one = (0,)
default_dfinfo = DataFlowInfo(subscr_one, subscr_all, subscr_all)

def dataflow_broadcast(arg_type, indices):
    if isinstance(arg_type, mytypes.ArrayType):
        dim = arg_type.ndim
        d = len(indices) - dim
        assert d >= 0
        subscr = indices[d:(d+dim)]
        if arg_type.shape:
            assert len(arg_type.shape) == dim
            return tuple(map(lambda i: 0 if arg_type.shape[i] == 1 else subscr[i], range(dim)))
        else:
            return subscr
    elif isinstance(arg_type, mytypes.NumType) or isinstance(arg_type, mytypes.BoolType):
        global subscr_one
        return subscr_one
    else:
        err.error('Cannot broadcast for:', arg_type)

def dataflow_ope1d_on_md(ndim, axis, indices):
    j = 0
    sub = []
    for i in range(ndim):
        if i == axis:
            global all_in_dim
            sub.append(all_in_dim)
        else:
            sub.append(indices[j])
            j += 1
    return tuple(sub)
