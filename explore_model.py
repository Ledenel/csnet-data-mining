from transformers import AutoTokenizer, AutoModel

from roberta_ast_label_pretrain import FinetuningRoberta, RobertaPretrain
from roberta_mask_pretrain import RobertaMaskPretrain, FinetuningMaskRoberta
import pandas as pd
import xarray as xa
from scipy import stats

if __name__ == '__main__':
    test_data = pd.read_pickle("data_cache/java_train_0.pkl")
    # model_all = RobertaPretrain.load_from_checkpoint(
    #     "pretrained_module/roberta_ast_label_pretrain_on_java_all-type_label/model.ckpt")
    #
    tokenizer = AutoTokenizer.from_pretrained(
        "huggingface/CodeBERTa-small-v1", resume_download=True)
    model = AutoModel.from_pretrained(
        "huggingface/CodeBERTa-small-v1", resume_download=True)

    encoded = tokenizer.encode_plus(test_data.docstring.array[0], test_data.code.array[0], return_tensors="pt")
    hiddens, first_output, attentions = model(**encoded, output_attentions=True)

    attn_array = xa.DataArray(data=[attn.detach().numpy() for attn in attentions],
                              dims=["layer", "batch", "head", "query_pos", "key_pos"])
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    df = attn_array.to_dataframe("attn")
    filter = df["attn"] > (1 / attn_array.sizes["key_pos"])
    df = df[filter]
    df["relative_pos"] = df.index.get_level_values("query_pos")
    df["relative_pos"] -= df.index.get_level_values("key_pos")
    df["query_word"] = df.index.get_level_values("query_pos").map(lambda x: tokens[x])
    df["key_word"] = df.index.get_level_values("key_pos").map(lambda x: tokens[x])
    df = df.droplevel("query_pos")
    df = df.droplevel("key_pos")
    df.groupby(["layer", "head"]).describe(percentiles=[0.01, 0.05, 0.2, 0.8, 0.95, 0.99]).to_csv("attn_describe.csv")
    df.sort_values(by="attn", ascending=False).to_csv("sorted_attn.csv")

