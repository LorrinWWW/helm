import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from trl.trainer import ConstantLengthDataset
import tqdm
import requests
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
    
    losses = []
    
    for i, item in enumerate(tqdm.tqdm(data, total=num_seqs)):
    
        if i >= num_seqs:
            break
        
        text = tokenizer.decode(item['input_ids'])

        for j in range(10):
            try:
                endpoint = 'http://127.0.0.1:8080'
                res = requests.post(endpoint, json={
                    "inputs": text,
                    "parameters":{
                        "max_new_tokens": 1, "do_sample": False, "logprobs": 1, "details": True, "decoder_input_details": True
                    }
                }, headers={})

                token_logprobs = [token['logprob'] for token in res.json()[0]['details']['prefill'][1:]]
                loss = -np.mean(token_logprobs)
                losses.append(loss)
                break
            except Exception as e:
                print(f'{e}: retry {i}')
                import time
                time.sleep(1)
            
        losses.append(loss)
    
    print(np.mean(losses), np.exp(np.mean(losses)))


if __name__ == '__main__':

    Fire(evaluate)