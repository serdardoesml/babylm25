# get_data.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datasets
import torch
import argparse
from functools import partial
from preprocessing import padding_collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sentence_splitter

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--gguf", type=str, default=None)
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--start_len", type=int, default=4)
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
def prep_dataset_autofill(examples, tokenizer, start_len=4):
    batch = {
        'input_ids':[],
        'attention_mask':[],
        'labels':[],
    }

    msg = """I will give you up to {} words at the start of a sentence. \
Complete the sentence, or if it is a full sentence, write the next sentence. \
Fix any spelling or punctuation mistakes. \
Write the whole sentence(s) and put it in this format: <sent> The sentence </sent>. \
\n\nHere is the start: {}"""

    for i in range(len(examples["text"])):
        src_start = examples["text"][i].split(" ")[:start_len]
        src_start = " ".join(src_start)
        messages = [
            {"role": "user", "content": msg.format(start_len, src_start)},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # src_start = #"Here is the completed sentence: " + src_start +

        src = tokenizer.encode(prompt + "<sent>", return_tensors="pt")[0]
        tgt = tokenizer.encode(examples["text"][i], return_tensors="pt")[0]

        batch['input_ids'].append(src)
        batch['labels'].append(tgt)
        batch['attention_mask'].append(torch.ones_like(src))

    return batch

def clean_output(outs):
    # return outs
    final_out = []

    def remove_chinese(text):
        return ''.join(char for char in text if not ('\u4e00' <= char <= '\u9fff'))

    for i in range(len(outs)):
        try:
            out = outs[i]
            # while out.startswith("!"):
            #     out = out[1:]

            # out = outs[i].split("Answer:")[-1] # remove everything before "Answer:"
            # out = outs[i].split("Here is the completed sentence:")[-1] # remove </s> and everything after
            out = out.split("<sent>")[-1] # remove </s> and everything after
            out = out.split("</sent>")[0] # remove </s> and everything after
            out = out.split("<|eot_id|>")[0].strip() # remove </s> and everything after
            # out = out.split("</s>")[0] # remove </s> and everything after
            # out = out.split("\"")[1].strip() # remove quotes, and stuff like "I hope this answer helped!"
            # out = sentence_splitter.split_text_into_sentences(out, language="en")[0]
            # out = remove_chinese(out)

            if out == "The sentence": # something wasn't generated properly, discard
                final_out.append("")
                continue
        except: # something wasn't generated properly, discard
            final_out.append("")
            continue
        final_out.append(out)

    return final_out


def main():
    args = parser.parse_args()
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=nf4_config, trust_remote_code=True, gguf_file=args.gguf)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dataset = datasets.load_dataset("text", data_files={"train": args.data_path})
    if args.debug:
        dataset['train'] = dataset['train'].select(range(100))

    dataset = dataset.map(partial(prep_dataset_autofill, tokenizer = tokenizer, start_len = args.start_len), batched = True, batch_size=1000, num_proc=10)

    dl = DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        collate_fn=partial(padding_collate_fn, skip_fields='text'),
    )

    print(dataset['train'][0])

    with torch.no_grad():
        outputs = []
        for batch in tqdm(dl):
            out = model.generate(batch['input_ids'].cuda(), 
                                 attention_mask=batch['attention_mask'].cuda(), 
                                 max_new_tokens=40, 
                                 do_sample=False,
                                #  num_beams=9,
                                #  num_beam_groups=3,
                                #  length_penalty=-1.0,
                                #  diversity_penalty=1.0,
                                #  repetition_penalty=1.6,
                                 temperature=1,
                                 tokenizer=tokenizer,
                                 stop_strings=[tokenizer.eos_token, "</sent>"]
                                )
            outs = tokenizer.batch_decode(out)
            
            true_outs = clean_output(outs)

            print(true_outs[0], f'\033[92m {batch["text"][0]} \033[0m')

            outputs.extend(true_outs)

    # clean empty outputs
    outputs = [out for out in outputs if out]

    # write to file
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write("\n".join(outputs))


if __name__ == "__main__":
    main()