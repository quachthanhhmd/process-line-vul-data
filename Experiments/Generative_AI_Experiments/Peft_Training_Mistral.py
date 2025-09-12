import numpy as np
from datasets import load_from_disk, load_metric, load_dataset
from transformers import AutoTokenizer, BertModel, AutoModelForSequenceClassification, T5ForSequenceClassification, LlamaTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch


def compute_metrics(eval_pred):
    metric_dict = {"Precision": load_metric("precision"),
                   "Recall": load_metric("recall"),
                   "Accuracy": load_metric("accuracy"),
                   "F1": load_metric("f1"),
                   "AUC": load_metric("roc_auc")}

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    result_dict = dict()
    for key, value in metric_dict.items():
        if key == 'AUC':
            result = value.compute(prediction_scores=predictions, references=labels)
        else:
            result = value.compute(predictions=predictions, references=labels)
        result_dict[key] = result
    return result_dict


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)


if __name__ == "__main__":
    print("Device Count", torch.cuda.device_count())
    MAX_LEN = 512
    ds = load_dataset("Partha117/Augmented_RealVul")
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = LlamaTokenizer.from_pretrained(model_id, add_prefix_space=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_ds = ds.map(preprocess_function, batched=True, num_proc=4)
    tokenized_ds.set_format("torch")
    tokenized_ds['train'] = tokenized_ds['train'].shuffle(seed=7)
    tokenized_ds = tokenized_ds['train'].train_test_split(test_size=0.1)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, id2label=id2label, load_in_4bit=True,
                                                               device_map='auto', label2id=label2id,
                                                               trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = model.config.eos_token_id
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="mistral_with_augmented_data",
        learning_rate=2e-7,
        per_device_train_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        do_train=True,
        do_eval=False,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=5,
        save_strategy="steps"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,

    )
    trainer.train()
