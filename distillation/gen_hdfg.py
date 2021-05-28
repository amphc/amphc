import sys
import re
import tokenize
from pathlib import Path
import pygraphviz
import networkx as nx

from lib2to3 import pygram, pytree
from lib2to3.pgen2 import driver, token
from lib2to3.pgen2.parse import ParseError
from lib2to3.pygram import python_symbols as syms
from lib2to3.pytree import Leaf, Node


class Cluster():
    registry = 0
    def __init__(self, nodes=None, var_expose="", loc=0):
        nodes = nodes if nodes is not None else []
        self.cid = Cluster.registry
        self.nodes = nodes
        self.var_expose = var_expose
        self.loc = loc
        self.call_expr = None
        Cluster.registry += 1

    def add_node(self, node):
        self.nodes.append(node)
        self.loc += node.loc

        # change node.cluster
        node.cluster = self

    def merge(self, other):
        self.nodes += other.nodes
        self.loc += other.loc

        for node in other.nodes:
            node.cluster = self


class DFGNode():
    registry = 0
    cache = {} # {rid: DFGNode}
    var_rid_dict = {} # {var_name: rid}
    clusters = {} # {cid: cluster}
    cluster_topo_order = []
    def __init__(
            self,
            variables=None,
            parents=None,
            cst_nodes=None,
            loc=0,
            var_expose="",
            is_const=False,
            is_compound=False
    ):
        variables = variables if variables is not None else []
        parents = parents if parents is not None else []
        self.rid = DFGNode.registry
        self.variables = variables
        # self.parents = parents
        self.parents = []
        self.add_parents(parents)
        self.children = []
        self.is_const = is_const
        self.is_compound = is_compound
        self.cst_nodes = [] # all cst nodes of statements that writes this variable (or varibales)
        self.add_cst_nodes(cst_nodes)
        self.loc = loc
        self.var_expose = var_expose
        self.cluster = None
        DFGNode.cache[self.rid] = self
        for var in self.variables:
            DFGNode.var_rid_dict[var] = self.rid
        DFGNode.registry += 1

    def __str__(self):
        return "id:%d vars: %s" % (self.rid, ','.join(self.variables))

    def __repr__(self):
        return str(self)

    def add_child(self, c):
        if c not in self.children:
            self.children.append(c)

    def add_cst_nodes(self, cst_nodes):
        for cst_node in cst_nodes:
            if self.is_compound:
                is_first_kw = True
                is_first_stat = True
                for flat_expr in flatten_some(cst_node.children):
                    if flat_expr.type == syms.simple_stmt:
                        if is_first_stat:
                            flat_expr.children[0].prefix = "    "
                            is_first_stat = False
                        else:
                            flat_expr.children[0].prefix = "        "
                    elif flat_expr.type == syms.for_stmt:
                        # nested loop
                        is_first_inner_kw = True
                        is_first_inner_stat = True
                        is_first_stat = True
                        for inner_flat_expr in flatten_some(flat_expr.children):
                            if inner_flat_expr.type == syms.simple_stmt:
                                # indent two tabs
                                if is_first_inner_stat:
                                    inner_flat_expr.children[0].prefix = "    "
                                    is_first_inner_stat = False
                                else:
                                    inner_flat_expr.children[0].prefix = "            "
                            else:
                                name = str(inner_flat_expr)
                                keywords = ["for", "if", "else"]
                                if name and not name.isspace():
                                    name = minimize_whitespace(str(inner_flat_expr))
                                if name in keywords:
                                    if is_first_inner_kw:
                                        inner_flat_expr.prefix = "        "
                                        is_first_inner_kw = False
                                    else:
                                        inner_flat_expr.prefix = "            "
                                        is_first_inner_stat = True
                    else:
                        name = str(flat_expr)
                        keywords = ["for", "if", "else"]
                        if name and not name.isspace():
                            name = minimize_whitespace(str(flat_expr))
                        if name in keywords:
                            if is_first_kw:
                                flat_expr.prefix = ""
                                is_first_kw = False
                            else:
                                flat_expr.prefix = "    "
                                is_first_stat = True
                self.cst_nodes.append(cst_node)
            else:
                cst_node.children[0].prefix = ""
                self.cst_nodes.append(cst_node)

    def __eq__(self, other):
        return self.rid == other.rid

    def __neq__(self, other):
        return self.rid != other.rid

    def insert_cst_to_head(self, cst_nodes):
        for node in cst_nodes:
            self.cst_nodes.insert(0, node)

    def add_variable(self, var):
        if var not in self.variables:
            self.variables.append(var)
        DFGNode.var_rid_dict[var] = self.rid

    def add_variables(self, variables):
        for var in variables:
            self.add_variable(var)

    def add_parent(self, p):
        if p.rid != self.rid and p.rid not in [pnode.rid for pnode in self.parents]:
            self.parents.append(p)

    def add_parents(self, ps):
        for p in ps:
            self.add_parent(p)

    def update_var_expose(self, new_var_expose):
        self.var_expose = new_var_expose

    def source(self):
        cst_strs = [str(node) for node in self.cst_nodes]
        return "\n".join(cst_strs)

    def to_json(self):
        return {
            'id':self.rid,
            'parents': [p.rid for p in self.parents],
            'children': [c.rid for c in self.children],
            'statments':self.source()
        }

    @classmethod
    def delete(cls, rid):
        for nvar, nrid in DFGNode.var_rid_dict.items():
            if nrid == rid:
                del DFGNode.var_rid_dict[nvar]
        cls.cache.pop(rid, None)

    @classmethod
    def clear(cls):
        cls.registry = 0
        cls.cache = {}
        cls.var_rid_dict = {}

    @classmethod
    def to_graph(cls):
        # G = pygraphviz.AGraph(directed=True)
        NxG = nx.DiGraph() # for topological sorting
        for nrid, cnode in DFGNode.cache.items():
            # G.add_node(cnode.rid)
            NxG.add_node(cnode.rid)
            # n = G.get_node(cnode.rid)
            # n.attr['label'] = "rid:"+str(cnode.rid)+"\nloc: "+str(cnode.loc)+"\nvars: "+','.join(cnode.variables)+"\nvar_expose:"+cnode.var_expose
            # if cnode.is_const:
            # 	n.attr['color'] = 'blue'
            # elif cnode.is_compound:
            # 	n.attr['color'] = 'green'
            for pn in cnode.parents:
                # G.add_edge(pn.rid, cnode.rid)
                NxG.add_edge(pn.rid, cnode.rid)

        for cid, cluster in DFGNode.clusters.items():
            node_ids = [node.rid for node in cluster.nodes]
            # G.add_subgraph(node_ids, name="cluster_"+cluster.var_expose, label="gen_"+cluster.var_expose, rank="same", color='red')
            nxSubgraph = NxG.subgraph(node_ids)
            cluster.topo_order = list(nx.topological_sort(nxSubgraph))

        # topological order of subgraph
        topo_node_rids = list(nx.topological_sort(NxG))
        DFGNode.cluster_topo_order = []
        for rid in topo_node_rids:
            node = DFGNode.cache[rid]
            if not node.is_const:
                if node.cluster.cid not in DFGNode.cluster_topo_order:
                    DFGNode.cluster_topo_order.append(node.cluster.cid)
                elif node.cluster.cid != DFGNode.cluster_topo_order[-1]:
                    DFGNode.cluster_topo_order.remove(node.cluster.cid)
                    DFGNode.cluster_topo_order.append(node.cluster.cid)

    # return G


class DFG():
    def __init__(self):
        self.constant_nodes = []
        self.param_nodes = []

    def update_children(self):
        for nrid, node in DFGNode.cache.items():
            for p in node.parents:
                p.add_child(node)

    def gen_dfg(self, lib2to3_node):
        for child in flatten_some(lib2to3_node.children):
            # TODO: import statement
            if child.type == syms.simple_stmt:
                # print("line", lineno, ":")
                maybe_expr = child.children[0]
                if maybe_expr.type == syms.expr_stmt:
                    expr = maybe_expr.children
                    self.check_and_create_node(expr, cst_node=child)
                else:
                    pass
            elif child.type == syms.if_stmt:
                if_start_lineno = child.get_lineno()
                if_end_lineno = 0

                compound_node = DFGNode(
                    variables=[],
                    parents=[],
                    cst_nodes=[child],
                    loc=0,
                    var_expose="",
                    is_compound=True
                )
                for flat_expr in flatten_some(child.children):
                    if flat_expr.type == syms.simple_stmt:
                        maybe_expr = flat_expr.children[0]
                        if maybe_expr.type == syms.expr_stmt:
                            expr = maybe_expr.children
                            self.check_and_create_node(expr, compound_node=compound_node)
                            if_end_lineno = max(if_end_lineno, expr[0].get_lineno())
                # TODO: what if blank lines are in function body?
                compound_node.loc += if_end_lineno - if_start_lineno + 1
            elif child.type == syms.for_stmt:
                for_start_lineno = child.get_lineno()
                for_end_lineno = 0

                compound_node = DFGNode(
                    variables=[],
                    parents=[],
                    cst_nodes=[child],
                    loc=0,
                    is_compound=True
                )
                for flat_expr in flatten_some(child.children):
                    if flat_expr.type == syms.simple_stmt:
                        maybe_expr = flat_expr.children[0]
                        if maybe_expr.type == syms.expr_stmt:
                            expr = maybe_expr.children
                            self.check_and_create_node(expr, compound_node=compound_node)
                            for_end_lineno = max(for_end_lineno, expr[0].get_lineno())
                    elif flat_expr.type == syms.for_stmt:
                        # TODO: remove hardcode two level!
                        for inner_flat_expr in flatten_some(flat_expr.children):
                            if inner_flat_expr.type == syms.simple_stmt:
                                maybe_expr = inner_flat_expr.children[0]
                                if maybe_expr.type == syms.expr_stmt:
                                    expr = maybe_expr.children
                                    self.check_and_create_node(expr, compound_node=compound_node)
                                    for_end_lineno = max(for_end_lineno, expr[0].get_lineno())
                # TODO: what if blank lines are in function body?
                compound_node.loc += for_end_lineno - for_start_lineno + 1

        self.update_children()

    def check_and_create_node(self, expr, cst_node=None, compound_node=None):
        # return lineno of expr
        if (
                isinstance(expr[0], Leaf)
                and expr[0].type == token.NAME
                and expr[1] == _eq
        ):
            node_name = minimize_whitespace(str(expr[0]))
        elif expr[1] == _eq:
            # expr[0] is a Node, for example, sliced list
            maybe_expr = expr[0].children[0]
            if isinstance(maybe_expr, Leaf) and maybe_expr.type == token.NAME:
                node_name = minimize_whitespace(str(maybe_expr))
        else:
            return

        rhs_expr = expr[2]
        parent_nodes = []
        if isinstance(rhs_expr, Leaf):
            if rhs_expr.type == token.NAME:
                rhs_expr_name = minimize_whitespace(str(rhs_expr))
                if rhs_expr_name in DFGNode.var_rid_dict:
                    parent_nodes.append(DFGNode.cache[DFGNode.var_rid_dict[rhs_expr_name]])
        else:
            # rhs_expr is a Node
            parent_nodes.extend(self.get_all_nodes_in_expr(rhs_expr.children))

        if compound_node is not None:
            if (node_name in DFGNode.var_rid_dict and
                    DFGNode.cache[DFGNode.var_rid_dict[node_name]] == compound_node):
                compound_node.add_parents(parent_nodes)
                compound_node.update_var_expose(node_name)
            elif node_name in DFGNode.var_rid_dict:
                node = DFGNode.cache[DFGNode.var_rid_dict[node_name]]
                # Merge node with compound node
                compound_node.add_variables(node.variables)
                # TODO: what if node is parent of other node?
                compound_node.add_parents(node.parents)
                compound_node.add_parents(parent_nodes)
                compound_node.insert_cst_to_head(node.cst_nodes)
                compound_node.loc += node.loc
                compound_node.update_var_expose(node_name)
                DFGNode.delete(node.rid)
            else:
                compound_node.add_variable(node_name)
                compound_node.add_parents(parent_nodes)
        else:
            # create new node for LHS variable
            if parent_nodes:
                if node_name in DFGNode.var_rid_dict:
                    DFGNode.cache[DFGNode.var_rid_dict[node_name]].add_parents(parent_nodes)
                else:
                    assign_node = DFGNode(
                        variables=[node_name],
                        parents=parent_nodes,
                        cst_nodes=[cst_node],
                        loc=1,
                        var_expose=node_name
                    )
            else:
                # no parent, constant
                assign_node = DFGNode(
                    variables=[node_name],
                    parents=parent_nodes,
                    cst_nodes=[cst_node],
                    loc=1,
                    is_const=True,
                    var_expose=node_name
                )

    def cluster(cls, threshold=10):
        # for each node, sort its parent nodes
        for rid in DFGNode.cache:
            DFGNode.cache[rid].parents.sort(key=lambda item: item.loc, reverse=True)

        # sort DFG.cache by node.loc
        DFGNode.cache = dict(
            sorted(DFGNode.cache.items(), key=lambda item: item[1].loc, reverse=True)
        )


        for rid, node in DFGNode.cache.items():
            if node.is_const:
                continue
            elif node.cluster != None:
                # check parent nodes
                for pnode in node.parents:
                    if node.cluster.loc > threshold:
                        break
                    if pnode.is_const:
                        continue

                    if pnode.cluster != None:
                        if node.cluster.loc + pnode.cluster.loc > threshold or len(
                                DFGNode.cache[DFGNode.var_rid_dict[pnode.cluster.var_expose]].children) > 1:
                            continue
                        else:
                            pcid = pnode.cluster.cid
                            node.cluster.merge(pnode.cluster)
                            del DFGNode.clusters[pcid]
                    else:
                        if len(pnode.children) > 1 or node.cluster.loc + pnode.loc > threshold:
                            continue
                        else:
                            node.cluster.add_node(pnode)
            else:
                new_cluster = Cluster(nodes=[node], var_expose=node.var_expose, loc=node.loc)
                DFGNode.clusters[new_cluster.cid] = new_cluster
                node.cluster = new_cluster

                # clustering
                for pnode in node.parents:
                    if pnode.is_const:
                        continue

                    if new_cluster.loc > threshold:
                        break

                    if pnode.cluster != None:
                        if new_cluster.loc + pnode.cluster.loc > threshold or len(
                                DFGNode.cache[DFGNode.var_rid_dict[pnode.cluster.var_expose]].children) > 1:
                            continue
                        else:
                            pcid = pnode.cluster.cid
                            new_cluster.merge(pnode.cluster)
                            del DFGNode.clusters[pcid]
                    else:
                        if len(pnode.children) > 1 or new_cluster.loc + pnode.loc > threshold:
                            continue
                        else:
                            new_cluster.add_node(pnode)

    def get_all_nodes_in_expr(self, children):
        for child in children:
            if isinstance(child, Leaf) and child.type == token.NAME:
                child_name = minimize_whitespace(str(child))
                if child_name in DFGNode.var_rid_dict:
                    yield DFGNode.cache[DFGNode.var_rid_dict[child_name]]
            else:
                for node in self.get_all_nodes_in_expr(child.children):
                    yield node

    def gen_code(self, input_node):
        output_node = Node(syms.file_input, children=[])

        # add import_stmt
        for child in flatten_some(input_node.children):
            if child.type != syms.simple_stmt:
                continue

            stmt = child.children[0]
            if stmt.type == syms.import_name or stmt.type == syms.import_from:
                output_node.children.append(child)

        for cid,cluster in DFGNode.clusters.items():
            func_name = Leaf(token.NAME, "gen_" + cluster.var_expose, prefix=" ")
            param_strs = []
            param_leaves = []
            for node in cluster.nodes:
                for pnode in node.parents:
                    if pnode.cluster is None or pnode.cluster.cid != cluster.cid:
                        if pnode.var_expose not in param_strs:
                            param_leaves.append(new(_comma))
                            param_leaves.append(Leaf(token.NAME, pnode.var_expose, prefix=" "))
                            param_strs.append(pnode.var_expose)
            # drop first comma, modify prefix for the first param
            if len(param_leaves) > 1:
                param_leaves = param_leaves[1:]
                param_leaves[0].prefix = ""

            parameters = Node(syms.parameters, children=[new(_lpar)]+param_leaves+[new(_rpar)])

            suite_children = [Leaf(token.NEWLINE, "\n")]
            for rid in cluster.topo_order:
                for cst_node in DFGNode.cache[rid].cst_nodes:
                    suite_children.append(Leaf(token.INDENT, "    "))
                    suite_children.append(new(cst_node))

            # add return_stmt
            suite_children.append(
                Node(
                    syms.return_stmt,
                    children=[
                        new(_return, prefix="    "),
                        Leaf(token.NAME, cluster.var_expose, prefix=" ")
                    ]
                )
            )
            suite = Node(syms.suite, children=suite_children)

            output_node.children.append(
                Node(
                    syms.funcdef,
                    children=[new(_def, prefix="\n"), func_name, parameters, new(_colon), suite]
                )
            )
            cluster.call_expr = [new(func_name), new(parameters)]
            output_node.children.append(Leaf(token.NEWLINE, "\n"))

        output_node.children.append(Leaf(token.NEWLINE, "\n"))

        # add all constants
        for rid,node in DFGNode.cache.items():
            if node.is_const:
                output_node.children += node.cst_nodes

        output_node.children.append(Leaf(token.NEWLINE, "\n"))

        # add stmt for generated functions
        for cid in DFGNode.cluster_topo_order:
            cluster = DFGNode.clusters[cid]
            expr_stmt = Node(
                syms.expr_stmt,
                children=[Leaf(token.NAME, cluster.var_expose), new(_eq, prefix=" ")]+cluster.call_expr
            )
            output_node.children.append(Node(syms.simple_stmt, children=[expr_stmt]))
            output_node.children.append(Leaf(token.NEWLINE, "\n"))

        return output_node


def extract_dfg(src):
    """Givn an input python file `src`, propagate types from caller to caller and save in `target_dir`"""
    src = Path(src)
    target_dir = Path('distill_output/dfg_transformation')
    target_dir.mkdir(parents=True, exist_ok=True)

    src = src.absolute()
    with tokenize.open(src) as src_buffer:
        src_contents = src_buffer.read()
        if src_contents == "":
            return
        src_encoding = src_buffer.encoding
        src_node = lib2to3_parse(src_contents)
        transformed_node = extract(src_node, src.name, target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    with open(target_dir / src.name, "w", encoding=src_encoding) as target_file:
        code_str = lib2to3_unparse(transformed_node)
        # print(code_str)
        target_file.write(code_str)


def extract(lib2to3_node, filename, target_dir):
    DFGNode.clear()
    dfg = DFG()
    dfg.gen_dfg(lib2to3_node)
    dfg.cluster(threshold=10)

    DFGNode.to_graph()
    # g = DFGNode.to_graph()
    # g.draw(target_dir / (filename+".png"), prog='dot')

    return dfg.gen_code(lib2to3_node)


def pop_param(params):
    """ (Use this func when params is not empty)
    Pops the parameter and the "remainder" (comma, default value).

    Returns a tuple of ('name', default) or (_star, 'name') or (_dstar, 'name').
    """
    default = None

    name = params.pop(0)
    if name in (_star, _dstar):
        default = params.pop(0)
        if default == _comma:
            return name, default

    try:
        if params:
            remainder = params.pop(0)
            if remainder == _eq:
                default = params.pop(0)
                remainder = params.pop(0)
            if remainder != _comma:
                raise ValueError(f"unexpected token: {remainder}")
    except IndexError:
        pass

    return name, default


def lib2to3_parse(src_txt):
    """Given a string with source, return the lib2to3 Node."""
    grammar = pygram.python_grammar_no_print_statement
    drv = driver.Driver(grammar, pytree.convert)
    if src_txt[-1] != "\n":
        nl = "\r\n" if "\r\n" in src_txt[:1024] else "\n"
        src_txt += nl
    try:
        result = drv.parse_string(src_txt, True)
    except ParseError as pe:
        lineno, column = pe.context[1]
        lines = src_txt.splitlines()
        try:
            faulty_line = lines[lineno - 1]
        except IndexError:
            faulty_line = "<line number missing in source>"
        raise ValueError(f"Cannot parse: {lineno}:{column}: {faulty_line}") from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])

    return result


def lib2to3_unparse(node):
    """Given a lib2to3 node, return its string representation."""
    code = str(node)
    return code


def flatten_some(children):
    """Generates nodes or leaves, unpacking bodies of try:except:finally: statements."""
    for node in children:
        if node.type in (syms.try_stmt, syms.suite):
            yield from flatten_some(node.children)
        else:
            yield node


def new(n, prefix=None):
    """lib2to3's AST requires unique objects as children."""

    if isinstance(n, Leaf):
        return Leaf(n.type, n.value, prefix=n.prefix if prefix is None else prefix)

    # this is hacky, we assume complex nodes are just being reused once from the
    # original AST.
    n.parent = None
    if prefix is not None:
        n.prefix = prefix
    return n


def minimize_whitespace(text):
    return re.sub(r"[\n\t ]+", " ", text, re.MULTILINE).strip().split()[-1]


_as = Leaf(token.NAME, "as", prefix=" ")
_colon = Leaf(token.COLON, ":")
_comma = Leaf(token.COMMA, ",")
_dot = Leaf(token.DOT, ".")
_dstar = Leaf(token.DOUBLESTAR, "**")
_eq = Leaf(token.EQUAL, "=", prefix=" ")
_lpar = Leaf(token.LPAR, "(")
_lsqb = Leaf(token.LSQB, "[")
_newline = Leaf(token.NEWLINE, "\n")
_none = Leaf(token.NAME, "None")
_rarrow = Leaf(token.RARROW, "->", prefix=" ")
_rpar = Leaf(token.RPAR, ")")
_rsqb = Leaf(token.RSQB, "]")
_star = Leaf(token.STAR, "*")
_ellipsis = Node(syms.atom, children=[new(_dot), new(_dot), new(_dot)])
_def = Leaf(token.NAME, "def")
_return = Leaf(token.NAME, "return")


if __name__ == "__main__":
    extract_dfg(sys.argv[1])
