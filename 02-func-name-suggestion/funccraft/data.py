from tree_sitter import Language, Parser
from pathlib import Path
from pprint import pprint
import datasets
import tree_sitter_python


PY_LANGUAGE = Language(tree_sitter_python.language())
PARSER = Parser(PY_LANGUAGE)


def remove_code_node(node, code, offset):
    start_byte, end_byte = node.start_byte, node.end_byte
    return code[:start_byte - offset] + code[end_byte - offset:], offset + node.end_byte - node.start_byte


def clean_code_node(node, code, offset):
    # Remove comments
    if node.type == 'comment':
        return remove_code_node(node, code, offset)

    # Remove strings (including docstrings)
    if node.type == 'string':
        parent = node.parent
        if parent and parent.type == 'expression_statement':
            return remove_code_node(node, code, offset)
        else:
            return code, offset
    
    # For all other nodes, including operators and variables, keep the text
    if len(node.children) == 0:
        return code, offset
    
    # Recursively process child nodes
    for child in node.children:
        code, offset = clean_code_node(child, code, offset)
    
    return code, offset


def process_function(code):
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
            func_body, _ = clean_code_node(node, modified_code, offset)
            result['my_func_without_comments'] = func_body
            break  # Handle only the first function

    return result


def prepare() -> datasets.Dataset:
    dataset = datasets.load_dataset('code-search-net/code_search_net', name='python', trust_remote_code=True, split='test')
    dataset = dataset.map(lambda example: process_function(example['whole_func_string']), batched=False)
    return dataset


def load_dataset(path: Path) -> datasets.Dataset:
    return datasets.load_from_disk(str(path))


def save_dataset(dataset: datasets.Dataset, path: Path) -> None:
    dataset.save_to_disk(str(path))
