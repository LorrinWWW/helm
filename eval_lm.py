import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from trl.trainer import ConstantLengthDataset
import tqdm
from fire import Fire


def evaluate(
    model_id: str,
    data_name: str,
    seq_length: int = 4096,
    num_seqs: int = 256,
):

    if data_name == 'c4':

        data = load_dataset('Jackmin108/c4-en-validation', split='validation')

    elif data_name == 'wikitext':

        data = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
    
    else:

        assert False

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    data = ConstantLengthDataset(
        tokenizer,
        data,
        dataset_text_field='text',
        infinite=False,
        shuffle=False,
        seq_length=seq_length,
        append_concat_token=False, 
        add_special_tokens=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, torch_dtype=torch.float16, 
        device_map="auto", low_cpu_mem_usage=True, attn_implementation="flash_attention_2",
    )
    
    losses = []
    
    for i, item in enumerate(tqdm.tqdm(data, total=num_seqs)):
    
        if i >= num_seqs:
            break
        
        text = tokenizer.decode(item['input_ids'])
    
        input_ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
    
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss.item()
            
        losses.append(loss)
    
    print(np.mean(losses), np.exp(np.mean(losses)))


if __name__ == '__main__':

    Fire(evaluate)