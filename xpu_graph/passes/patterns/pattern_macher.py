from xpu_graph.passes.patterns.pattern import Pattern
import torch.fx as fx
import re
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_element_in_list(element, my_list):
    return element in my_list

class PatMatcher(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        if self.rewriter(gm):
            changed = True

        return changed
    def rewriter(self, gm: fx.GraphModule) -> bool:
        raise NotImplementedError

class Node:
    def __init__(self, name, type, idd):
        self.args  = []
        self.users = []
        self.type  = type
        self.idd   = idd
        if type == "constant":
            self.vals = [float(name)]
            self.names = ["c"]
        else:
            self.names = name if isinstance(name, list) else [name]

    def target(self):
        return '/'.join(self.names) + f"_{self.idd}"

    def __repr__(self):
        argnames = ",".join([item.target() for item in self.args])
        usernames = ",".join([item.target() for item in self.users])
        return f"{self.target()}[{self.type}]({argnames})->({usernames})"


class Graph:
    def __init__(self):
        self.nodes = []
        self.inputs = []
        self.outputs = []
        self.variables = {}

    def __repr__(self):
        output = [f"Graph: {len(self.nodes)} nodes, {len(self.inputs)} inputs, {len(self.outputs)} outputs"]
        output.append(f"Inputs:")
        for node in self.inputs:
            output.append(f"\t{node}")

        output.append("")
        output.append(f"Body:")
        for node in self.nodes:
            output.append(f"\t{node}")

        output.append("")
        output.append(f"Outputs:")
        for node in self.outputs:
            output.append(f"\t{node}")
        return "\n".join(output)
    def add_node(self, name, args, outputs, idd):
        node = []
        node_args = []
        node_users = []
        if name[0] == "placeholder":
            node = Node(outputs, "placeholder", idd)
            self.inputs.append(node)
        else:
            node = Node(name, "module", idd)
            for arg in args:
                if (arg != "?" and arg not in self.variables) and not is_number(arg):
                    is_input = False
                    for input in self.inputs:
                        if is_element_in_list(arg, input.names):
                            is_input = True
                            node_args.append(input)
                            break
                    if is_input:
                        continue
                    else:
                        raise KeyError(f"Undefined argument: [{arg}]")

                if arg == "?":
                    new_node = Node("?", "placeholder", idd)
                    new_node.users.append(node)
                    self.inputs.append(new_node)
                    node_args.append(new_node)
                    continue
                if is_number(arg):
                    new_node = Node(arg, "constant", idd)
                    new_node.users.append(node)
                    self.inputs.append(new_node)
                    node_args.append(new_node)
                    continue
                parent = self.variables[arg]
                node_args.append(parent)
                try:
                    i = parent.users.index(arg)
                    parent.users[i] = node
                except ValueError:
                    parent.users.append(node)

            for user in outputs:
                if user == "?":
                    new_node = Node("?", "output", idd)
                    new_node.args.append(node)
                    self.outputs.append(new_node)
                    node_users.append(new_node)
                    continue
                    
                node_users.append(user)
                self.variables[user] = node
            
            node.args = node_args
            node.users = node_users
            self.nodes.append(node)

class Lexer:
    def __init__(self, pattern):
        
        # Compile the extraction regular expression.
        extract_operator_signature = re.compile("([\W\w]+)=([\W\w]+)\(([\W\w]*)\)")
        # Remove spaces and split patterns by the break line.
        lines = [item.strip() for item in pattern.replace(" ", "").split("\n")]

        # Parsing patterns by lexical analyzer.
        self.pattern  = pattern
        self.lines    = lines
        self.graph    = Graph()
        for iline, line in enumerate(lines):
            if line.startswith("#") or line == "":
                continue

            op_sig = extract_operator_signature.findall(line)
            
            assert len(op_sig) == 1, f"Unexpected line: {line}. The valid symbol is: name(input_argument, output_argument)"
            outputs, op_names, inputs = op_sig[0]
            self.graph.add_node(op_names.split("/"), inputs.split(","), outputs.split(","), iline)

class Matcher():
    def __init__(self, gm, pat):
        self.matched = []
        self.gm  = gm
        self.modules = dict()

        # match
        self.lexer = Lexer(pat)
        all_matched_pairs = []
        for node in self.gm.graph.nodes:
            all_matched_pairs.extend(self._try_to_match(node))
        self.matched = all_matched_pairs
    def _try_to_match(self, anchor):
        matched_paths = []
        params_stack = [[[anchor], self.lexer.graph.nodes[0]]]
        while len(params_stack) > 0:
            path, condition = params_stack.pop()
            if condition.type == "output":
                matched_paths.append(path[:-1])
                continue

            anchor = path[-1]
            if not self._match(condition, anchor):
                continue
                
            all_outputs_is_placeholder = all([item.type == "output" for item in condition.users])
            for i, output_user in enumerate(anchor.users):
                if all_outputs_is_placeholder:
                    params_stack.append([path + [output_user], condition.users[0]])
                else:
                    params_stack.append([path + [output_user], condition.users[i]])

        return matched_paths
    def print_matchs(self):
        pass
    # replace some subgraph to new
    def replace(self, new_graph_fn=None):
        for i, subgraph in enumerate(self.matched):
            new_graph_fn(self, i, subgraph)

        self.recompile()
        return self
    
    def recompile(self):
        self.traced.graph.lint()
        self.traced.recompile()
        self.modules = dict(self.traced.named_modules())
        return self    