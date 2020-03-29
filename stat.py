from dataset import CodeSearchDatasetLoader
import pandas as pd
from collections import defaultdict, Counter
from attributedict.collections import AttributeDict
from tqdm import tqdm

if __name__ == "__main__":
    all_ds = CodeSearchDatasetLoader().get()
    counter_all = defaultdict(Counter)
    for sample in tqdm(all_ds):
        for field, typ, text in sample.text_ctx:
            counter_all[(sample.language, field, typ)][text] += 1
    
    attrib_list = []
    for key, cnt in tqdm(counter_all.items()):
        row = AttributeDict()
        row.lang, row.field, row.typ = key
        cnt: Counter
        row.unique_count = len(cnt.keys())
        row.all_count = sum(cnt.values())
        for i, (val, count) in enumerate(cnt.most_common(3)):
            row[f'top{i+1}_text'] = val
            row[f'top{i+1}_count'] = count
    
    pd.DataFrame(attrib_list).to_csv("field_context_stats.csv")

        
