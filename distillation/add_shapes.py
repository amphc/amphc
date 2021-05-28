import argparse
import ast
import os
import random
import sys
from pathlib import Path
from types import FrameType
from unittest.mock import patch

from monkeytype.config import DefaultConfig
from monkeytype.tracing import CallTracer
from monkeytype.tracing import get_type
from pytype.pyi import parser
from pytype.pytd.pytd import GenericType
from pytype.pytd.pytd import NamedType
from pytype.pytd.pytd_visitors import PrintVisitor
from pytype.pytd.pytd_visitors import Visitor

PYTYPE_STUB_DIRECTORY = "distill_output/pytype/pyi/"
OUTPUT_STUB_DIRECTORY = "distill_output/monkeytype/pyi/"
BASE_DIRECTORY = "distill_output/monkeytype"
mt_db_path = os.path.abspath("distill_output/monkeytype/monkeytype.sqlite3")
os.environ["MT_DB_PATH"] = mt_db_path


class RevisedPrintVisitor(PrintVisitor):
    def _RequireImport(self, module, name=None):
        """Register that we're using name from module.

        Args:
        module: string identifier.
        name: if None, means we want 'import module'. Otherwise string identifier
        that we want to import.
        """
        # This is a hack, some of the dimension info is polluting the AST. need to purge
        if not "[" in module:
            self.imports[module].add(name)

    def VisitGenericType(self, node):
        """Convert a generic type to a string."""
        parameters = node.parameters
        if self._IsEmptyTuple(node):
            parameters = ("()",)
        elif self._NeedsTupleEllipsis(node):
            parameters += ("...",)
        elif self._NeedsCallableEllipsis(self.old_node):
            parameters = ("...",) + parameters[1:]
        return self.MaybeCapitalize(node.base_type) + "[" + "][".join(parameters) + "]"


class AppendToConst(Visitor):
    """finds the root const we're looking for and adding shape information"""

    def __init__(self, arg, shape):
        super(AppendToConst, self).__init__()
        self.arg = arg
        self.shape = shape
        self.entered_const = False
        self.entered_generic = False

    def EnterConstant(self, node):
        if node.name != self.arg:
            return False
        self.entered_const = True

    def LeaveConstant(self, node):
        self.entered_const = False

    def EnterGenericType(self, node):
        self.entered_generic = True

    def LeaveGenericType(self, node):
        self.entered_generic = False

    def VisitGenericType(self, node):
        if self.entered_const:
            return node.Replace(parameters=node.parameters + (NamedType(self.shape),))
        else:
            return node

    def VisitNamedType(self, node):
        if self.entered_const:
            if self.entered_generic:
                return node
            else:
                return GenericType(node, (NamedType(self.shape),))
        return node


class PappaTracer(CallTracer):
    """modified call tracer that will try to patch pytype pyi files with shape and dtype info as it is found"""

    def __init__(self, logger, max_typed_dict_size, code_filter, sample_rate):
        super().__init__(logger, max_typed_dict_size, code_filter, sample_rate)
        self.stub_ast_dict = {}
        self.code_ast_dict = {}
        self.call_back_lineno = 0
        self.root_path = os.getcwd()

    def write(self):
        Path(self.root_path + "/" + OUTPUT_STUB_DIRECTORY).mkdir(parents=True, exist_ok=True)
        for k, v in self.stub_ast_dict.items():
            new_pyi_filename = self.root_path + "/" + OUTPUT_STUB_DIRECTORY + k + ".pyi"
            with open(new_pyi_filename, "w") as out_file:
                out_file.write(v.Visit(RevisedPrintVisitor()))

    def find_and_add_to_pyi_file(self, file_path, arg, shape):
        _, file_name = os.path.split(file_path)
        file_base = os.path.splitext(file_name)[0]
        ast_to_visit = self.stub_ast_dict.get(file_base, None)
        if not ast_to_visit:
            pyi_path = self.root_path + "/" + PYTYPE_STUB_DIRECTORY + file_base + ".pyi"
            try:
                self.stub_ast_dict[file_base] = parser.parse_file(pyi_path, (3, 7, 7))
            except Exception as exc:
                print(
                    "The pytype stub file is missing for one of the calls, did you make stub files?"
                )
                sys.exit(str(exc))
            ast_to_visit = self.stub_ast_dict.get(file_base)
        self.stub_ast_dict[file_base] = ast_to_visit.Visit(AppendToConst(arg, shape))

    def get_code_ast(self, file_path):
        _, file_name = os.path.split(file_path)
        file_base = os.path.splitext(file_name)[0]
        ast_to_visit = self.code_ast_dict.get(file_base, None)
        if not ast_to_visit:
            try:
                with open(file_path) as f:
                    self.code_ast_dict[file_base] = ast.parse(f.read(), filename=file_path)
            except Exception as exc:
                print(
                    "The code file is missing for one of the calls, did you rearrange files?"
                )
                sys.exit(str(exc))
            ast_to_visit = self.code_ast_dict.get(file_base)
        return ast_to_visit

    def handle_call(self, frame: FrameType) -> None:
        if self.sample_rate and random.randrange(self.sample_rate) != 0:
            return
        func = self._get_func(frame)
        if func is None:
            return
        # ensure we're at module-level calls
        if frame.f_back.f_code.co_name != "<module>":
            return
        self.call_back_lineno = frame.f_back.f_lineno

    def handle_return(self, frame: FrameType, arg) -> None:
        typ = get_type(arg, max_typed_dict_size=self.max_typed_dict_size)
        if frame.f_back.f_lineno != self.call_back_lineno:
            return
        try:
            typename = typ.__name__
        except AttributeError:
            return
        if typename in {"ndarray", "DataFrame"}:
            shape = ", ".join([str(s) for s in arg.shape])
            # TODO: This can be done once and indexed
            ast_to_walk = self.get_code_ast(frame.f_code.co_filename)
            for item in ast.walk(ast_to_walk):
                if not isinstance(item, ast.Assign):
                    continue
                if not isinstance(item.value, ast.Call):
                    continue
                # TODO: handle multiple assignments
                if isinstance(item.targets[0], ast.Subscript):
                    continue
                line_to_compare = item.value.lineno
                if len(item.value.args) != 0:
                    line_to_compare = item.value.args[-1].lineno
                if self.call_back_lineno == line_to_compare:
                    self.find_and_add_to_pyi_file(
                        frame.f_code.co_filename, item.targets[0].id, shape
                    )


class MyConfig(DefaultConfig):
    def sample_rate(self):
        return 1


def add_shapes(file_to_run, global_vars, local_vars):
    print("updating stub files with dtype and shape")
    if not os.path.exists(BASE_DIRECTORY):
        os.makedirs(BASE_DIRECTORY)
    my_config = MyConfig()
    logger = my_config.trace_logger()
    tracer = PappaTracer(
        logger=logger,
        max_typed_dict_size=my_config.max_typed_dict_size(),
        code_filter=my_config.code_filter(),
        sample_rate=my_config.sample_rate(),
    )
    sys.setprofile(tracer)
    path_name, file_name = os.path.split(file_to_run)
    pure_preamble = "def pure(f):\n    f.pure = True\n    return f\n" # currently unused
    with open(file_to_run, "r") as source_file:
        code = compile(source_file.read(), file_name, "exec")
    with patch('sys.argv', [file_name]):
        cwd = os.getcwd()
        if path_name:
            sys.path.append(path_name)
            os.chdir(path_name)
        exec(code, global_vars, local_vars)
        os.chdir(cwd)
    sys.setprofile(None)  # remove the tracer
    tracer.write()
    print("stub files updated successfully")


def main(global_vars=None, local_vars=None):
    arg_parser = argparse.ArgumentParser(
        description="dynamically adds datatype and shape info to pytype annotations"
    )
    arg_parser.add_argument("file_to_run")
    args = arg_parser.parse_args()
    add_shapes(args.file_to_run, global_vars, local_vars)


if __name__ == "__main__":
    main(globals(), locals())
