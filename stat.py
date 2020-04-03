from dataset import CodeSearchDatasetLoader
import dataset as ds
import pandas as pd
from collections import defaultdict, Counter
from attributedict.collections import AttributeDict
from tqdm import tqdm
import multiprocessing

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        ds.global_pool = pool
        all_ds = CodeSearchDatasetLoader().get(force_lazy=True)
        counter_all = defaultdict(Counter)
        for sample in tqdm(all_ds):
            for parent_field, parent, field, typ, text in sample.text_ctx:
                counter_all[(sample.language, parent_field, parent, field, typ)][text] += 1
        
        attrib_list = []
        for key, cnt in tqdm(counter_all.items()):
            row = AttributeDict()
            row.lang, row.parent_field, row.parent, row.field, row.typ = key
            cnt: Counter
            row.unique_count = len(cnt.keys())
            row.all_count = sum(cnt.values())
            for i, (val, count) in enumerate(cnt.most_common(3)):
                row[f'top{i+1}_text'] = val
                row[f'top{i+1}_count'] = count
            attrib_list.append(row)
        
        pd.DataFrame(tqdm(attrib_list)).to_csv("stats/field_context_l2_stats.csv")
    ds.global_pool = None

        
