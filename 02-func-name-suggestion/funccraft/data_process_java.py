from tree_sitter import Language, Parser, Node
import tree_sitter_java

JAVA_LANGUAGE = Language(tree_sitter_java.language())
PARSER = Parser(JAVA_LANGUAGE)


def remove_code_node(node, code, offset):
    start_byte, end_byte = node.start_byte, node.end_byte
    return code[:start_byte - offset] + code[end_byte - offset:], offset + node.end_byte - node.start_byte


def clean_code_node(node, code, offset):
    # Remove comments
    if node.type == 'line_comment' or node.type == 'block_comment':
        return remove_code_node(node, code, offset)
    
    # For all other nodes, including operators and variables, keep the text
    if len(node.children) == 0:
        return code, offset
    
    # Recursively process child nodes
    for child in node.children:
        code, offset = clean_code_node(child, code, offset)
    
    return code, offset


def process_function_java(code):
    tree = PARSER.parse(code.encode('utf-8'))
    root_node = tree.root_node

    result = {'my_func_name': None, 'my_func_with_comments': None, 'my_func_without_comments': None}

    for node in root_node.children:
        if node.type == 'method_declaration':
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
            func_body, _ = clean_code_node(node, modified_code, offset)
            result['my_func_without_comments'] = func_body
            break  # Handle only the first function

    return result