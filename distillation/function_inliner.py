import sys
import imp
import ast
import astunparse
import copy
from collections import OrderedDict
from pathlib import Path
from astmonkey.transformers import ParentChildNodeTransformer


class ParamReplacer(ast.NodeTransformer):
    def __init__(self, param_mapping):
        self.mapping = param_mapping

    def visit_Name(self, node):
        return self.mapping.get(node.id, node) or node


class ReturnToAssign(ast.NodeTransformer):
    def __init__(self, targets):
        self.return_val = None
        self.is_return_replaced = False
        self.targets = targets

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            self.return_val = node.value.id
            # drop return
            return None
        else:
            self.is_return_replaced = True
            return ast.Assign(targets=self.targets, value=node.value)


class BaseFunctionHandler(object):
    def replace_params_with_objects(self, target_node, inline_func, call_object):
        """
        target_node is some AST object, could be the return value of a function we are inlining.
        We need to inspect its parameters and create a dictionary then use ParamReplacer to replace
        all instances of those parameters with the local references to the objects being passed in
        """
        args = inline_func.args
        default_offset = len(args.args) - len(args.defaults)

        arg_mapping = OrderedDict()
        for idx, arg in enumerate(arg for arg in args.args if not arg.arg == "self"):
            arg_mapping[arg.arg] = None
            if idx >= default_offset:
                arg_mapping[arg.arg] = args.defaults[idx - default_offset]

            if len(call_object.args) > idx:
                arg_mapping[arg.arg] = call_object.args[idx]

        for keyword in call_object.keywords:
            arg_mapping[keyword.arg] = keyword.value

        if len([arg for arg in args.args if arg.arg == "self"]):
            # Ok, get the name of "self" (the instance of the class we are using)
            new_mapping = OrderedDict({"self": call_object.func.value})
            new_mapping.update(arg_mapping)
            arg_mapping = new_mapping

        return ParamReplacer(arg_mapping).visit(target_node)

    def return_to_assign(self, targets, func_body_replaced):
        return_to_assign_transformer = ReturnToAssign(targets)
        transformed_node = return_to_assign_transformer.visit(func_body_replaced)
        if return_to_assign_transformer.is_return_replaced:
            return transformed_node
        else:
            # replace return_value with targets[0]
            name_mapping = {return_to_assign_transformer.return_val: targets[0]}
            return ParamReplacer(name_mapping).visit(transformed_node)


class FunctionHandler(BaseFunctionHandler):
    def inline(self, node, func_to_inline):
        # node is an Assign node that call the function, func_to_incline is the function definition

        # Step 1: replace parameters with passed objects for the whole function body
        func_to_inline_copy = copy.deepcopy(func_to_inline)
        func_replaced = self.replace_params_with_objects(
            func_to_inline_copy,
            func_to_inline,
            node.value
        )

        # Step 2: replace return expr with Assign or change return val to target (if directly return single val)
        inline_body_node = self.return_to_assign(node.targets, func_replaced).body

        return inline_body_node


def getFunctionName(func):
    if hasattr(func, "id"):
        return func.id
    else:
        if isinstance(func, ast.Attribute):
            return func.attr
        # Might be a class item
        return func.name


class SSA_Formater(ast.NodeTransformer):
    def __init__(self):
        self.assign_count = {} # {name: num_of_times_assigned_value}
        self.naming_mapping = {} # {original_name: cur_name_node}, only contains variable has been renamed

    def visit_If(self, node):
        return node

    def visit_Assign(self, node):
        # TODO: what if there are more than one target in the Assign node
        if isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id

            # examine value: if contains renamed varible, replaced.
            new_value_node = ParamReplacer(self.naming_mapping).visit(node.value)

            # rename target variable if applicable
            new_target_node = node.targets
            if target_name in self.assign_count:
                new_name = target_name + '_' + str(self.assign_count[target_name])
                new_target_node = [ast.Name(id=new_name, ctx=ast.Store())]
                self.naming_mapping[target_name] = ast.Name(id=new_name, ctx=ast.Load())
                self.assign_count[target_name] += 1
            else:
                self.assign_count[target_name] = 1

            return ast.Assign(targets=new_target_node, value=new_value_node)

        elif isinstance(node.targets[0], ast.Subscript):
            # replace both target and value with current naming mapping (don't update target naming)
            return ParamReplacer(self.naming_mapping).visit(node)


class FunctionInliner(ast.NodeTransformer):
    def __init__(self, functions_to_inline):
        self.inline_funcs = functions_to_inline

    def visit_Assign(self, node):
        result_node = node
        if isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = getFunctionName(func)
            if func_name in self.inline_funcs:
                func_to_inline = self.inline_funcs[func_name]
                result_node = FunctionHandler().inline(node, func_to_inline)

        return result_node


class FunctionRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        return None


class IfMainRemover(ast.NodeTransformer):
    def visit_If(self, node):
        if node.test.left.id == '__name__' and node.test.comparators[0].s == '__main__':
            return node.body
        else:
            return node


class InlineMethodLocator(ast.NodeVisitor):
    def __init__(self):
        self.functions = {}

    def visit_FunctionDef(self, node):
        # if any(filter(lambda d: d.id == "inline", node.decorator_list)):
        func_name = getFunctionName(node)
        self.functions[func_name] = node


def inline_funcs(pythonfile):
    #transform and ouput a new file
    ast_node_inlined = None
    with open(pythonfile) as pf:
        ast_node = ast.parse(pf.read())
        # print(ast.dump(ast_node))
        tree = ParentChildNodeTransformer().visit(ast_node)
        function_disoverer = InlineMethodLocator()
        function_disoverer.visit(tree)
        print("found funcs:", function_disoverer.functions)
        ast_node_inlined = FunctionInliner(function_disoverer.functions).visit(tree)
        ast_node_inlined = FunctionRemover().visit(ast_node_inlined)
        ast_node_inlined = IfMainRemover().visit(ast_node_inlined)

        ast_node_inlined_ssa = SSA_Formater().visit(ast_node_inlined)
    # ast_node_inlined_ssa = AssignInIf_Adjuster().visit(ast_node_inlined_ssa)

    target_dir = Path('distill_output/function_inline/')
    target_dir.mkdir(parents=True, exist_ok=True)
    src = Path(pythonfile)
    with open(target_dir / src.name, "w") as out_file:
        out_file.write(astunparse.unparse(ast_node_inlined))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pythonfile', help='The python file to be analyzed (full/relative path)')
    # parser.add_argument('-w', action='store_true', help='a *_purified.py file will be written')

    args = parser.parse_args()
    inline_funcs(args.pythonfile)
