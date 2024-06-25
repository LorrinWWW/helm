import torch
import numpy as np
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from trl.trainer import ConstantLengthDataset
import tqdm
from fire import Fire
import sys
from quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize
import os


@torch.no_grad()
def calibrate_func(model, tokenizer, batch_size, batches):
    samples = batch_size * batches
    #cal_dataset = load_dataset("Jackmin108/c4-en-validation", split=["validation"])[0]
    cal_dataset = load_dataset("lambada", split=["validation"])[0]
    model.eval()
    total = 0
    for batch in cal_dataset.iter(batch_size=batch_size):
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        model(input_ids, attention_mask=attention_mask)
        total += input_ids.size(0)
        if total >= samples:
            break


def evaluate(
    model_id: str,
    data_name: str,
    quant_param: str,
    seq_length: int = 4096,
    num_seqs: int = 256,
):
    # HF_token = "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"
    # os.environ["HUGGINGFACE_TOKEN"] = HF_token
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
        # WARNING this may fail if your GPU does not have enough memory
    model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map='auto'
        )
    if quant_param == 'weight':
        quantize(model, weights=qfloat8)
        freeze(model)
        print('weight quantization done')
    if quant_param == 'activation':
        quantize(model, weights=qfloat8, activations=qfloat8)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        batch_size = 32
        batches = 4
        torch.backends.cudnn.enable = True
        torch.backends.cudnn.benchmark = True
        with Calibration(momentum=0.8):
            calibrate_func(model, tokenizer, batch_size, batches)
        freeze(model) 
        print('activation quantization done')
    losses = []
    print('evaluating ...')
    for i, item in enumerate(tqdm.tqdm(data, total=num_seqs)):
    
        if i >= num_seqs:
            break
        
        text = tokenizer.decode(item['input_ids'])
    
        input_ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to(model.device)
    
        with torch.no_grad():
            loss = model(input_ids, labels=input_ids).loss.item()
            
        losses.append(loss)
    
    print(np.mean(losses), np.exp(np.mean(losses)))


# call the evaluate function with command line arguments
if __name__ == '__main__':
    model_id = sys.argv[1]
    data_name = sys.argv[2]
    quant_param = sys.argv[3]
    evaluate(model_id, data_name, quant_param)
    