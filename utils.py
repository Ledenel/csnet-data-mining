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

def fetch_snakemake_from_latest_run(script_path):
    import glob
    import re
    import os
    _, file_name = os.path.split(script_path)
    script_regex = re.compile(rf'^.snakemake/scripts/tmp.+\.{file_name}$')
    snakemake_mark = re.compile(r"^#+ .+ #+$")
    script_versions = glob.glob(".snakemake/scripts/*.py")
    script_versions = [script for script in script_versions if script_regex.match(script)]
    script_versions.sort(key=lambda path: os.stat(path).st_mtime)
    latest_script_path = script_versions[-1]
    with open(latest_script_path, "r") as f:
        for line in f:
            if snakemake_mark.match(line.strip()): #snakemake header start
                break
        header_content = ""
        for line in f:
            if snakemake_mark.match(line.strip()): #origin script start
                break
            else:
                header_content += f"{line}\n"
        print("HEADER:\n", header_content)
        prepared_global_context = {"__file__":script_path}
        prepared_local_context = {}
        context = exec(header_content, prepared_global_context, prepared_local_context)
        snakemake = prepared_local_context['snakemake']
    return snakemake