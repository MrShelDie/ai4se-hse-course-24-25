from tree_sitter import Language, Parser, Node
import tree_sitter_python


PY_LANGUAGE = Language(tree_sitter_python.language())
PARSER = Parser(PY_LANGUAGE)


def remove_code_node(node: Node, code: str, offset: int) -> tuple[str, int]:
    """
    Removes the code corresponding to the given node,
    adjusting the offset of the remaining code relative to the original.

    Args:
        node (Node): The node whose code needs to be removed.
        code (str): The original code.
        offset (int): The offset at which the code is currently located relative to the original code.

    Returns:
        tuple[str, int]: A tuple containing the new code and the new offset.
    """
    start_byte, end_byte = node.start_byte, node.end_byte
    return code[:start_byte - offset] + code[end_byte - offset:], offset + node.end_byte - node.start_byte


def remove_comments_node(node: Node, code: str, offset: int) -> tuple[str, int]:
    """
    Removes comments (including docstrings) from the given code based on the current node.

    Args:
        node (Node): The current node in the syntax tree.
        code (str): The original code from which comments are to be removed.
        offset (int): The current offset in the code.

    Returns:
        tuple[str, int]: A tuple containing the code without comments and the updated offset.
    """
    # Remove comments    
    if node.type == 'comment':
        return remove_code_node(node, code, offset)

    # Remove docstrings
    if node.type == 'string':
        parent = node.parent
        if parent and parent.type == 'expression_statement':
            return remove_code_node(node, code, offset)
        else:
            return code, offset
    
    # For leaf nodes, keep the text
    if len(node.children) == 0:
        return code, offset
    
    # Recursively process child nodes
    for child in node.children:
        code, offset = remove_comments_node(child, code, offset)
    
    return code, offset


def process_function_python(code: str) -> dict[str, str]:
    """
    Process a Python function by extracting its name, replacing its name with a special token, and removing comments.

    Args:
        code (str): The original code of the Python function.

    Returns:
        dict[str, str]: A dictionary with keys 'my_func_name', 'my_func_with_comments', 'my_func_without_comments'
        containing:
            the original function name,
            the code with the special token <extra_id_0> and comments,
            the code with the special token <extra_id_0> and without comments respectively.
    """
    tree = PARSER.parse(code.encode('utf-8'))
    root_node = tree.root_node

    result = {'my_func_name': None, 'my_func_with_comments': None, 'my_func_without_comments': None}

    for node in root_node.children:
        if node.type == 'function_definition':
            # Retrieve the function name
            func_name_node = node.child_by_field_name('name')
            func_name = code[func_name_node.start_byte : func_name_node.end_byte]
            result['my_func_name'] = func_name

            # Change the function name to '<extra_id_0>'
            start_byte, end_byte = func_name_node.start_byte, func_name_node.end_byte
            token = '<extra_id_0>'
            modified_code = (
                code[:start_byte]
                + token
                + code[end_byte:]
            )
            result['my_func_with_comments'] = modified_code

            # Remove comments
            offset = len(func_name) - len(token)
            func_body, _ = remove_comments_node(node, modified_code, offset)
            result['my_func_without_comments'] = func_body
            break  # Handle only the first function

    return result