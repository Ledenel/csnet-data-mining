from transformers import AutoTokenizer, AutoModel

from roberta_ast_label_pretrain import FinetuningRoberta, RobertaPretrain
from roberta_mask_pretrain import RobertaMaskPretrain, FinetuningMaskRoberta
import pandas as pd

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
    output, attentions = model(**encoded, output_attentions=True)
    print(output)
