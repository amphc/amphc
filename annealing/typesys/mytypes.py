import typed_ast.ast3 as ast
import util.error as err

operational_order = { 'bool': 0, 'int': 1, 'float': 2, 'complex': 3, 'num': 4, 'any': 5 }

### Base types (not visible to users) ###
class MyType:
    name = 'my'
    order = -1
    limited = set()
    operators = {}
    attributes = ()

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if self.order == -1 or other.order == -1:
            err.error('Type comparison is not supported between:', self.name, 'and', other.name)
        return self.order < other.order

    def __gt__(self, other):
        if self.order == -1 or other.order == -1:
            err.error('Type comparison is not supported between:', self.name, 'and', other.name)
        return self.order > other.order

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __str__(self):
        return self.name

class IterableType(MyType):
    name = 'iterable'

    def __init__(self, etype):
        if self.limited and etype.name not in self.limited:
            err.error('Invalid element type:', etype.name + ', which is not in', self.name, 'type\'s valid set:', self.limited, ')')

        self.etype = etype

    def __eq__(self, other):
        return MyType.__eq__(self, other) and self.etype == other.etype

    def __str__(self):
        return '%s(%s)' % (self.name, self.etype)

class CollectiveType(MyType):
    name = 'collective'


### Unknown type ###
class AnyType(MyType):
    name = 'any'
    order = operational_order[name]


### Numeric scalar types ###
class NumType(MyType):
    name = 'num'
    order = operational_order[name]

    def __init__(self, width = -1):
        if self.limited and width not in self.limited:
            err.error('Invalid bit-width:', width, '(not in', self.name, 'valid set:', self.limited, ')')

        self.width = width

    def __eq__(self, other):
        return MyType.__eq__(self, other) and self.width == other.width

    def __lt__(self, other):
        if other.order == -1:
            err.error('Type comparison is not supported between:', self.name, 'and', other.name)
        return self.order < other.order or self.order == other.order and self.width < other.width

    def __gt__(self, other):
        if other.order == -1:
            err.error('Type comparison is not supported between:', self.name, 'and', other.name)
        return self.order > other.order or self.order == other.order and self.width > other.width

    def __str__(self):
        if self.width == -1:
            return self.name
        else:
            return self.name + str(self.width)

class IntType(NumType):
    name = 'int'
    order = operational_order[name]
    limited = {32, 64}

class FloatType(NumType):
    name = 'float'
    order = operational_order[name]
    limited = {32, 64}

class ComplexType(NumType):
    name = 'complex'
    order = operational_order[name]
    limited = {64, 128}


### Non-numeric scalar types ###
class BoolType(MyType):
    name = 'bool'
    order = operational_order[name]

class StrType(MyType):
    name = 'str'


### Intrepydd collective types ###
'''
Collection whose elements have unique type (etype).
 - Dense and sparse arrays:
  -- Array: N-dimensional, dense, ordered and changeable
  -- SparseMat: 2-dimensional, sparse, ordered and changeable

 - Allows duplicate members:
  -- UTuple (Uniform Tuple): Ordered and unchangeable
  -- UList (Uniform List): Ordered and changeable

 - No duplicate members:
  -- USet (Uniform Set): Unordered and unindexed
  -- UDict (Uniform Dictionary): Unordered, changeable and indexed
'''
class ArrayType(IterableType):
    name = 'Array'
    limited = {'int', 'float', 'complex', 'num', 'bool'}
    # WORKING: with libbase
    operators = { ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mult', ast.Div: 'div', ast.MatMult: 'matmult', ast.Pow: 'power' }
    attributes = ('add', 'sub', 'mult', 'multiply', 'div', 'divide', 'matmult', 'transpose')

    def __init__(self, etype, ndim = -1, shape = (), module = 'numpy'):
        if shape and ndim != len(shape):
            err.error('Unmatched # dimensions: ndim =', ndim, 'vs. len(shape) =', len(shape))

        IterableType.__init__(self, etype)
        self.ndim = ndim
        self.shape = shape
        self.module = module

    def __eq__(self, other):
        # Todo: Consider shape for comparison?
        return IterableType.__eq__(self, other) and self.ndim == other.ndim and self.module == other.module

    def __str__(self):
        if self.shape:
            return '%s(%s, %s, %s, %s)' % (self.name, self.etype, self.ndim, self.shape, self.module)
        else:
            return '%s(%s, %s, %s)' % (self.name, self.etype, self.ndim, self.module)

class SparseMatType(IterableType):
    name = 'SparseMat'
    limited = {'int', 'float', 'complex', 'num', 'bool'}
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC

    def __init__(self, etype):
        IterableType.__init__(self, etype)
        self.ndim = 2

    def __eq__(self, other):
        return IterableType.__eq__(self, other) and self.ndim == other.ndim

class UTupleType(IterableType):
    name = 'UTuple'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC

class UListType(IterableType):
    name = 'UList'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC

class USetType(IterableType):
    name = 'USet'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC
    limited = {'int', 'float', 'complex', 'num', 'bool', 'str', 'tuple', 'UTuple'}

class UDictType(IterableType):
    name = 'UDict'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC
    limited = {'int', 'float', 'complex', 'num', 'bool', 'str', 'tuple', 'UTuple'}

    def __init__(self, ktype, vtype):
        IterableType.__init__(self, ktype)  # Type of key (element)
        self.vtype = vtype                  # Type of value

    def __eq__(self, other):
        return IterableType.__eq__(self, other) and self.vtype == other.vtype

    def __str__(self):
        return '%s(%s, %s)' % (self.name, self.etype, self.vtype)


### Python collective types ###
class TupleType(CollectiveType):
    name = 'tuple'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC

class ListType(CollectiveType):
    name = 'list'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC

class SetType(CollectiveType):
    name = 'set'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC

class DictType(CollectiveType):
    name = 'dict'
    operators = {}   # WORKING: TBC
    attributes = ()  # WORKING: TBC
