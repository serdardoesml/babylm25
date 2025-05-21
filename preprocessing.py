# preprocessing.py
import torch

def group_texts(examples, max_length, pad_token_id):
    grouped = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }

    for i in range(len(examples['input_ids'])):
        for j in range(0, len(examples['input_ids'][i]), max_length):
            if j + max_length > len(examples['input_ids'][i]):
                grouped['input_ids'].append(examples['input_ids'][i][j:] + [pad_token_id] * (max_length - (len(examples['input_ids'][i]) - j)))
                grouped['attention_mask'].append(examples['attention_mask'][i][j:] + [0] * (max_length - (len(examples['input_ids'][i]) - j)))
                grouped['labels'].append(examples['labels'][i][j:] + [pad_token_id] * (max_length - (len(examples['input_ids'][i]) - j)))
            else:
                grouped['input_ids'].append(examples['input_ids'][i][j : j + max_length])
                grouped['attention_mask'].append(examples['attention_mask'][i][j : j + max_length])
                grouped['labels'].append(examples['labels'][i][j : j + max_length])

    return grouped


def tokenize(examples, tokenizer, input_field):
    encoded = {
        'input_ids': [], 
        'attention_mask': [], 
        'labels': []}
    for i in range(len(examples[input_field])):
        src = examples[input_field][i]
        src_tok = [tokenizer.bos_token_id] + tokenizer.encode(src, add_special_tokens=False) + [tokenizer.eos_token_id]
        encoded['input_ids'] += [src_tok[:-1]]
        encoded['labels'] += [src_tok[1:]]
        attention_mask = torch.ones(len(src_tok)-1, dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]
    return encoded



def padding_collate_fn(batch, max_len=2048, skip_fields=[]):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        if key in skip_fields:
            padded_batch[key] = []
            continue
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if "labels" in key:
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key in skip_fields: 
                padded_batch[key].append(batch[i][key]) 
                continue
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])

    return padded_batch