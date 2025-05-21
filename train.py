# train.py
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, set_seed
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM, DebertaV2Tokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, load_from_disk
import torch
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", type=str, default="")
parser.add_argument("--valid_data", type=str, default="data/even.dev")
parser.add_argument("--max_seq_len", type=int, default=64)
parser.add_argument("--model_path", type=str, default="microsoft/deberta-v3-base")
parser.add_argument("--model_type", type=str, default="encoder")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--tokenizer", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--grad_acc", type=int, default=1)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--cpus", type=int, default=64)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_only", action="store_true", help="Evaluate only")
parser.add_argument("--debug", action="store_true", help="Activates debug mode")
parser.add_argument("--preprocessed", action="store_true", help="Use preprocessed data")
parser.add_argument("--rope", action="store_true", help="Use RoPE")
# parser.add_argument("--custom", action="store_true", help="Use custom model")

GPT2_CONFIG = {
    'hidden_size': 768,
    'intermediate_size': 3072,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
}

# def prep_dataset(examples, tokenizer):
#     batch = {
#         'input_ids':[],
#         'attention_mask':[],
#         'labels':[],
#     }
#     for i in range(len(examples["text"])):
#         txt = examples["text"][i]
#         enc = tokenizer.encode(txt, add_special_tokens=True)
#         batch['input_ids'].append(enc)
#         batch['labels'].append(enc)
#         batch['attention_mask'].append([1]*len(enc))


#     return batch

def prep_dataset(examples, tokenizer):
    batch = {
        'input_ids':[],
        'attention_mask':[],
        'labels':[],
    }
    for i in range(len(examples["text"])):
        txt = examples["text"][i]
        enc = tokenizer.encode(txt, add_special_tokens=True)
        batch['input_ids'].append(enc)
        batch['labels'].append(enc)
        batch['attention_mask'].append([1]*len(enc))
    return batch

def formatting_prompts_func(examples, tokenizer):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# def formatting_cute(examples, tokenizer):
#     formatted = {"text" : []}
#     for i in range(len(examples["prompt"])):
#         text = [examples["prompt"][i], "Answer: \"" + examples["answer"][i]]
        
#         texts = [tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt = False) + " Answer: \"" for prompt in prompts]
#     return formatted

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.flatten()
    labels = labels.flatten()
    mask = labels != -100
    labels = labels[mask]
    predictions = predictions[mask]

    correct = labels == predictions
    accuracy = correct.sum() / float(len(correct))
    return {"acc": accuracy}


def main():
    args = parser.parse_args()
    set_seed(args.seed)

    if not args.debug:
        # Unsloth detects other gpus when using nohup, even when setting this outside of the script.
        # Setting it here doesn't affect anything, but allows unsloth to run. 
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


    tokenizer = DebertaV2Tokenizer(args.tokenizer)
    config = AutoConfig.from_pretrained(args.model_path)
    if args.rope:
        config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
        for key in GPT2_CONFIG:
            setattr(config, key, GPT2_CONFIG[key])
        config.num_key_value_heads = config.num_attention_heads

    config.vocab_size = tokenizer.vocab_size
    config.max_position_embeddings = 1024


    # if args.custom:
    #     from custom_models import CustomLlamaForCausalLM
    #     model = CustomLlamaForCausalLM(config, tokenizer)
    if args.model_type == "encoder":
        model = AutoModelForMaskedLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_config(config)

    if not args.preprocessed:
        dataset = load_dataset('text', data_files = {'train': args.train_data, 'validation': args.valid_data})
        # dataset = dataset.map(partial(prep_dataset, tokenizer = tokenizer), batched = True)

        # dataset['test'] = load_dataset("leukas/cute", split="del_char")

    else:
        dataset = load_from_disk(args.dataset)


    if args.model_type == "encoder":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            seed=args.seed
            )
    else:
        # add labels = input_ids
        data_collator = DataCollatorForLanguageModeling(
            mlm=False,
            tokenizer = tokenizer
            )


    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = dataset['train'],
        eval_dataset = dataset['validation'],
        data_collator = data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        args = SFTConfig(
            label_names=["labels"],
            dataset_num_proc = args.cpus,
            packing = True,
            eval_packing=True,
            max_seq_length = args.max_seq_len,
            dataset_text_field = "text",
            eval_strategy="steps",
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_acc,
            warmup_ratio = 0.05,
            num_train_epochs = args.epochs,
            # max_steps = 60,
            learning_rate = args.lr,
            fp16 = False,
            bf16 = True,
            logging_steps = args.logging_steps,
            eval_steps = args.eval_steps,
            save_steps = args.save_steps,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = args.seed,
            output_dir = f"{args.output_path}" if args.output_path else None,
            report_to = "none",
            eval_accumulation_steps=1,
            include_for_metrics=["inputs"] if args.eval_only else [],
            max_grad_norm=1,
        ),
    )

    # if args.model_type == "decoder":
    #     trainer.train_dataset = trainer.train_dataset.map(lambda x: {"labels": x["input_ids"]}, batched = True)

    # trainer = train_on_responses_only(
    #     trainer,
    #     instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    #     response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    # )

    # print(tokenizer.decode(trainer.train_dataset[5]["input_ids"], remove_special_tokens=False))

    # labels = trainer.train_dataset[5]["labels"]
    # labels = torch.clamp(torch.tensor(labels), min=0)
    # print(tokenizer.decode(labels))    # tok_min = 1000
    # tok_max = 0
    # for i in range(len(trainer.train_dataset)):
    #     tok_min = min(min(trainer.train_dataset[i]["input_ids"]), tok_min)
    #     tok_max = max(max(trainer.train_dataset[i]["input_ids"]), tok_max)
    # print(tok_min, tok_max)

    # print("max len:", len(trainer.train_dataset[5]["input_ids"]))
    # print(tokenizer.convert_ids_to_tokens(trainer.train_dataset[5]["input_ids"]))

    if not args.eval_only:
        trainer_stats = trainer.train()

    # output = trainer.predict(trainer.eval_dataset)
    # print(output.metrics)
    # if args.output_path:
    #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    #     write_output_to_file(output, tokenizer, args.output_path)
    
if __name__ == "__main__":
    main()