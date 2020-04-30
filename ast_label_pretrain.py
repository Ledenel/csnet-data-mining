from parsing.sitter_lang import get_parser
from tree_sitter import Node, TreeCursor
import seqtools as sq


def node_cursor_iter(cursor):
    yield cursor.copy()
    if cursor.goto_first_child():
        yield from node_cursor_iter(cursor)
        while cursor.goto_next_sibling():
            yield from node_cursor_iter(cursor)
        cursor.goto_parent()


def self_chain(cur):
    field_name = cur.current_field_name()
    yield field_name, cur.copy()
    while cur.goto_parent():
        yield field_name, cur.copy()
        field_name = cur.current_field_name()


def node_asts(ast):
    return [
        cursor.node
        for cursor in node_cursor_iter(ast.walk())
    ]


def _sub_code_pieces(ast: Node, code):
    return [
        code[node.start_byte:node.end_byte]
        for node in node_asts(ast)
    ]


def _sub_labels(ast):
    return [
        [
            (name, parent.node.type)
            for name, parent in self_chain(cursor)
        ]
        for cursor in node_cursor_iter(ast.walk())
    ]


def seq_from_code_ast(_seq_dict):
    _code_bytes = _seq_dict["code_bytes"]
    _asts = _seq_dict["asts"]
    sub_code_pieces = sq.smap(_sub_code_pieces, _asts, _code_bytes)
    # sub_asts = sq.smap(_sub_labels, _asts)
    sub_labels = sq.smap(_sub_labels, _asts)
    type_label = sq.smap(
        lambda lbs: [[x[1] for x in labels] for labels in lbs],
        sub_labels
    )
    combined_label = sq.smap(
        lambda lbs: [[f"{x[0]}-{x[1]}" for x in labels] for labels in lbs],
        sub_labels
    )

    _dict_all = locals()
    _dict_return = {k: v for k, v in _dict_all.items()
                    if not k.startswith("_")}
    # print(_dict_return.keys())
    return _dict_return
