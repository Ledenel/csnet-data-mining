import pandas as pd

def get_columns_by_all_func(df, func, head_sample=5):
    if head_sample is not None:
        df_head = df.head(head_sample)
    else:
        df_head = df
    func_filter = df_head.applymap(func).all()
    return df[df.columns[func_filter]]

def count_describe_column(series, topk=3):
    value_counts = series.value_counts()
    value_counts_tops = value_counts.sort_values(ascending=False)[:topk]
    result_global = pd.Series({
        "total": value_counts.sum(),
        "unique": len(value_counts),
    })
    result_count_describe = value_counts.describe(percentiles=[.001,.01,.05,.2,.8,.95,.99,.999])
    del result_count_describe["count"]
    result_tops = [
        pd.Series({
            f"top{i+1}_item": term,
            f"top{i+1}_count": cnt,
        }) for i, (term, cnt) in enumerate(value_counts_tops.iteritems())
    ]
    result = pd.concat([result_global, result_count_describe] + result_tops)
    result.name = series.name
    return result

def value_counts_describe(df, head_sample=5):
    explodable_columns = get_columns_by_all_func(df, pd.api.types.is_list_like, head_sample=head_sample)
    exploded_columns = [df[x].explode() for x in explodable_columns]
    hashable_columns = get_columns_by_all_func(df, pd.api.types.is_hashable, head_sample=head_sample)
    hashable_columns = [df[x] for x in hashable_columns]
    describe_cols = [count_describe_column(x) for x in exploded_columns + hashable_columns]
    all_describe = pd.concat(describe_cols, axis="columns").T
    print(all_describe)
    max_eq_min = all_describe["min"] == all_describe["max"]
    return all_describe[~max_eq_min]
