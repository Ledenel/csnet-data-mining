
def node_cursor_iter(cursor):
    yield cursor
    if cursor.goto_first_child():
        yield from node_cursor_iter(cursor)
        while cursor.goto_next_sibling():
            yield from node_cursor_iter(cursor)
        cursor.goto_parent()

import copy

def full_context(cur):
    field_name = cur.current_field_name()
    while cur.goto_parent():
        yield f"{cur.node.type}:{field_name}"
        field_name = cur.current_field_name()

def identifier_cursors(tree, source):
    cursor_root = tree.walk()
    for cur in node_cursor_iter(cursor_root):
        node = cur.node
        if "identifier" in node.type:
            yield source[node.start_byte:node.end_byte].decode('utf-8'), cur.copy()

            #TODO publish introduced binding copy().

def identifier_context(tree, source):
    for name, cur_node in identifier_cursors(tree, source):
        node = cur_node.node
        parent_node_cur = cur_node.copy()
        parent_node_cur.goto_parent()
        yield "{}:{} is from {}".format(
            node.type, source[node.start_byte:node.end_byte].decode('utf-8'), 
            "->".join(reversed(list(full_context(cur_node)))),
        )
            #do not use origin cursor, node.walk() drop parents. copy cursor or recover state 

def parent_chain(cur):
    field_name = cur.current_field_name()
    while cur.goto_parent():
        yield field_name, cur
        field_name = cur.current_field_name()
            

def node_iter(tree):
    cursor = tree.walk()
    return (c.node for c in node_cursor_iter(cursor))


language = raw_sample['language']
if language.startswith('python'):  # In some datasets, we use 'python-2.7' and 'python-3'
    language = 'python'
parser = get_parser(language)
parser_lang = get_language(language)
code_raw = raw_sample['code'].encode("utf-8")
code_ast = parser.parse(code_raw)
vocab_description = per_code_language_metadata[language]['token_vocab']
word_to_int, int_to_word = vocab_description.word_vocab, vocab_description.inverse_word_vocab
# identifier_predicate = parser_lang.query("((identifier))")
# identifiers = identifier_predicate.captures(code_ast.root_node)
all_nodes = list(node_iter(code_ast))

ident_ctx = list(identifier_context(code_ast, code_raw))
range_ident_parent_map = defaultdict(list)
ident_parent_type_map = defaultdict(list)
for name, cur in identifier_cursors(code_ast, code_raw):
    for i, (field_name, cur_parent) in enumerate(parent_chain(cur.copy())):
        parent = cur_parent.node
        key = (parent.type, field_name, parent.start_byte, parent.end_byte)
        extra_key = (parent.type, None, parent.start_byte, parent.end_byte)
        parent_type = parent.type
        val = (i,name,cur.node.start_byte)
        range_ident_parent_map[key].append(val)
        if key != extra_key:
            range_ident_parent_map[extra_key].append(val)
        
        