from typing import Iterable
from tree_sitter import TreeCursor

def node_cursor_iter(cursor) -> Iterable[TreeCursor]:
    yield cursor
    if cursor.goto_first_child():
        yield from node_cursor_iter(cursor)
        while cursor.goto_next_sibling():
            yield from node_cursor_iter(cursor)
        cursor.goto_parent()
