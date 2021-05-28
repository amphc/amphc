import typed_ast.ast3 as ast
import util.error as err
import typesys.mytypes as mytypes
from libbase.libfuncs import *


def get_func_info(func_name, module_names):
    if module_names:
        main = module_names[0]
        sub = module_names[1:] if len(module_names) > 1 else []
        info = module_to_info(main, sub)
        if info and func_name in info.funcinfo:
            return info.funcinfo[func_name]
    else:
        if func_name in funcinfo:
            return funcinfo[func_name]
    return None

# Intrepydd built-in functions
funcinfo = {
    'int': IntInfo(),
    'len': IntInfo(),
    'range': RangeInfo(),
    'arange': ArangeInfo(),
    'linspace': LinSpaceInfo(),
    'empty': NewArrayInfo(),
    'zeros': NewArrayInfo(),
    'ones': NewArrayInfo(),
    'randn': RandnInfo(),
    'copy': CopyInfo(),
    'asarray': AsArrayInfo(),
    'ravel': RavelInfo(),
    'squeeze': SqueezeInfo(),
    'reshape': ReshapeInfo(),
    'tile': TileInfo(),
    'transpose': TransposeInfo(),
    'sqrt': UnaryInfo(),
    'exp': UnaryInfo(),
    'log': UnaryInfo(),
    'ceil': UnaryInfo(),
    'conj': UnaryInfo(),
    'add': ArithmeticInfo(ast.Add()),
    'sub': ArithmeticInfo(ast.Sub()),
    'mult': ArithmeticInfo(ast.Mult()),
    'multiply': ArithmeticInfo(ast.Mult()),
    'power': ArithmeticInfo(ast.Pow()),
    'div': ArithmeticInfo(ast.Div()),
    'divide': ArithmeticInfo(ast.Div()),
    'matmult': MatMultInfo(),
    'matmul': MatMultInfo(),
    'fft': FFTInfo(),
    'ifft': FFTInfo(),
    'fftshift': FFTShiftInfo(),
}


# Todo: improve this test implementations
imported_modules = {}
v2m = {}
def import_module(module, var):
    if module not in imported_modules:
        if module == 'numpy':
            imported_modules[module] = NumpyMdInfo()
        elif module == 'cupy':
            imported_modules[module] = CupyMdInfo()
        elif module == 'ctypes' or module == 'sys' or module == 'time':
            err.verbose(4, 'Not related to AST opt; ignore:', module)
            return False
        else:
            err.error('Unsupported module:', module)
    v2m[var] = module
    return True

def var_to_module(var, info = False):
    module = v2m[var] if var in v2m else None
    if info:
        return imported_modules[module] if module else None
    else:
        return module

def module_to_var(module, get_all = False):
    if get_all:
        var = [v for v in v2m if v2m[v] == module]
    else:
        var = next((v for v in v2m if v2m[v] == module), None)
    return var

def module_to_info(module, sub = []):
    info = imported_modules[module] if module in imported_modules else None
    if info:
        for s in sub:
            if s in info.moduleinfo:
                info = info.moduleinfo[s]
            else:
                return None
    return info

class ModuleInfo:
    name = ''
    mname = ''
    funcinfo = {}
    moduleinfo = {}

class NumpyRandomMdInfo(ModuleInfo):
    name = 'numpy.random'
    mname = 'numpy'
    funcinfo = {
        'randn': RandnInfo(mname),
        }

class NumpyFFTMdInfo(ModuleInfo):
    name = 'numpy.fft'
    mname = 'numpy'
    funcinfo = {
        'fft': FFTInfo(mname),
        'ifft': FFTInfo(mname),
        'fftshift': FFTShiftInfo(mname),
        }

class NumpyMdInfo(ModuleInfo):
    name = 'numpy'
    mname = name
    funcinfo = {
        'arange': ArangeInfo(mname),
        'linspace': LinSpaceInfo(mname),
        'empty': NewArrayInfo(mname),
        'zeros': NewArrayInfo(mname),
        'ones': NewArrayInfo(mname),
        'copy': CopyInfo(mname),
        'asarray': AsArrayInfo(mname),
        'ravel': RavelInfo(mname),
        'squeeze': SqueezeInfo(mname),
        'reshape': ReshapeInfo(mname),
        'tile': TileInfo(mname),
        'transpose': TransposeInfo(mname),
        'sqrt': UnaryInfo(mname),
        'exp': UnaryInfo(mname),
        'log': UnaryInfo(mname),
        'ceil': UnaryInfo(mname),
        'conj': UnaryInfo(mname),
        'add': ArithmeticInfo(ast.Add(), mname),
        'multiply': ArithmeticInfo(ast.Mult(), mname),
        'power': ArithmeticInfo(ast.Pow(), mname),
        'matmul': MatMultInfo(mname),
        }
    moduleinfo = {
        'random': NumpyRandomMdInfo(),
        'fft': NumpyFFTMdInfo(),
        }
    dtypes = {
        'int32': mytypes.IntType(32),
        'int64': mytypes.IntType(64),
        'float32': mytypes.FloatType(32),
        'float64': mytypes.FloatType(64),
        'complex64': mytypes.ComplexType(64),
        'complex128': mytypes.ComplexType(128),
        'bool': mytypes.BoolType(),
        }
    atypes = {
        'ndarray': mytypes.ArrayType(mytypes.NumType(), -1, module = mname)
        }
    const = {
        'pi': mytypes.FloatType(64)
        }

class CupyRandomMdInfo(ModuleInfo):
    name = 'cupy.random'
    mname = 'cupy'
    funcinfo = {
        'randn': RandnInfo(mname),
        }

class CupyFFTMdInfo(ModuleInfo):
    name = 'cupy.fft'
    mname = 'cupy'
    funcinfo = {
        'fft': FFTInfo(mname),
        'ifft': FFTInfo(mname),
        'fftshift': FFTShiftInfo(mname),
        }

class CupyMdInfo(ModuleInfo):
    name = 'cupy'
    mname = name
    funcinfo = {
        'arange': ArangeInfo(mname),
        'linspace': LinSpaceInfo(mname),
        'empty': NewArrayInfo(mname),
        'zeros': NewArrayInfo(mname),
        'ones': NewArrayInfo(mname),
        'copy': CopyInfo(mname),
        'asnumpy': CopyToHostInfo(mname),
        'asarray': AsArrayInfo(mname),
        'ravel': RavelInfo(mname),
        'squeeze': SqueezeInfo(mname),
        'reshape': ReshapeInfo(mname),
        'tile': TileInfo(mname),
        'transpose': TransposeInfo(mname),
        'sqrt': UnaryInfo(mname),
        'exp': UnaryInfo(mname),
        'log': UnaryInfo(mname),
        'ceil': UnaryInfo(mname),
        'conj': UnaryInfo(mname),
        'add': ArithmeticInfo(ast.Add(), mname),
        'multiply': ArithmeticInfo(ast.Mult(), mname),
        'power': ArithmeticInfo(ast.Pow(), mname),
        'matmul': MatMultInfo(mname),
        }
    moduleinfo = {
        'random': CupyRandomMdInfo(),
        'fft': CupyFFTMdInfo(),
        }
    dtypes = { # For Intrepydd array annotations, e.g., cupy.complex128[:1, :]
        'int32': mytypes.IntType(32),
        'int64': mytypes.IntType(64),
        'float32': mytypes.FloatType(32),
        'float64': mytypes.FloatType(64),
        'complex64': mytypes.ComplexType(64),
        'complex128': mytypes.ComplexType(128),
        'bool': mytypes.BoolType(),
        }
    atypes = {
        'ndarray': mytypes.ArrayType(mytypes.NumType(), -1, module = mname)
        }
