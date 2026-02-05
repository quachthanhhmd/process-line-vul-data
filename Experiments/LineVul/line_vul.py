import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix
import torch
import gc
import pickle
from transformers import TrainingArguments, Trainer,RobertaForSequenceClassification,RobertaTokenizer
from transformers import EarlyStoppingCallback
from sklearn.utils import class_weight
import ray
import argparse
from tqdm import tqdm
import os
from os.path import join, exists
import re
import torch.nn as nn
from collections import Counter
import random
import pickle
import json
import hashlib

from collections import defaultdict

ray.init(_plasma_directory="/tmp")
torch.cuda.empty_cache()
gc.collect()

def getMD5(s):
    hl = hashlib.md5()
    hl.update(s.encode("utf-8"))
    return hl.hexdigest()

def removeComments(text):
    """ remove c-style comments.
        text: blob of text with comments (can include newlines)
        returns: text with comments removed
    """
    pattern = r"""
                            ##  --------- COMMENT ---------
           //.*?$           ##  Start of // .... comment
         |                  ##
           /\*              ##  Start of /* ... */ comment
           [^*]*\*+         ##  Non-* followed by 1-or-more *'s
           (                ##
             [^/*][^*]*\*+  ##
           )*               ##  0-or-more things which don't start with /
                            ##    but do end with '*'
           /                ##  End of /* ... */ comment
         |                  ##  -OR-  various things which aren't comments:
           (                ##
                            ##  ------ " ... " STRING ------
             "              ##  Start of " ... " string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^"\\]       ##  Non "\ characters
             )*             ##
             "              ##  End of " ... " string
           |                ##  -OR-
                            ##
                            ##  ------ ' ... ' STRING ------
             '              ##  Start of ' ... ' string
             (              ##
               \\.          ##  Escaped char
             |              ##  -OR-
               [^'\\]       ##  Non '\ characters
             )*             ##
             '              ##  End of ' ... ' string
           |                ##  -OR-
                            ##
                            ##  ------ ANYTHING ELSE -------
             .              ##  Anything other char
             [^/"'\\]*      ##  Chars which doesn't start a comment, string
           )                ##    or escape
    """
    regex = re.compile(pattern, re.VERBOSE|re.MULTILINE|re.DOTALL)
    noncomments = [m.group(2) for m in regex.finditer(text) if m.group(2)]
    noncomments="".join(noncomments)
    return noncomments

    
class Dataset(torch.utils.data.Dataset):    
    def __init__(self, encodings, labels=None):          
        self.encodings = encodings        
        self.labels = labels
     
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item     
    def __len__(self):
        return len(self.encodings["input_ids"])
    
def create_token_chunks_vulnerable_samples(code_statements,all_special_ids,vulnerable_line_numbers):
    i=0
    samples,labels=[],[]
    while i<len(code_statements):
        tokens=[]
        k=i
        while i<len(code_statements):
            modified_input_ids=[]
            for j in range(len(code_statements[i])):
                if code_statements[i][j] not in all_special_ids:
                    modified_input_ids.append(code_statements[i][j])

            if len(tokens)+len(modified_input_ids)<=510:
                tokens.extend(modified_input_ids)
            else:
                break
            i+=1
        flag=False
        samples.append(removeComments(tokenizer.decode(tokens)))
        for line_number in vulnerable_line_numbers:
            if int(line_number) in range(k,i):
                flag=True
                break
        if flag:
            labels.append(1)
        else:
            labels.append(0)
    return samples,labels

def read_file_label(sample,tokenizer):
    label=1 if sample["vulnerable_line_numbers"] else 0
    all_special_ids=tokenizer.all_special_ids
    source_code=sample["processed_func"].split("\n")
    inputs,labels=[],[]
    flaw_lines, flaw_line_indices = [], []
    
    # Get metadata from sample, safely handle missing keys
    flaw_line_val = sample.get("flaw_line", "")
    flaw_line_index_val = sample.get("flaw_line_index", "")

    if label==1:
        samples,mixed_labels=create_token_chunks_vulnerable_samples(tokenizer(source_code)["input_ids"],all_special_ids,sample["vulnerable_line_numbers"].split(","))
        inputs.extend(samples)
        labels.extend(mixed_labels)
    else:
        input_id=tokenizer(removeComments("".join(source_code)))["input_ids"]
        modified_input_ids=[]
        for i in range(len(input_id)):
            if input_id[i] not in all_special_ids:
                modified_input_ids.append(input_id[i])
        for i in range(0,len(modified_input_ids),510):
            inputs.append(tokenizer.decode(modified_input_ids[i:i+510]))
            labels.append(label)
    
    # Propagate metadata for each chunk derived from this sample
    for _ in range(len(inputs)):
        flaw_lines.append(flaw_line_val)
        flaw_line_indices.append(flaw_line_index_val)

    return inputs,labels,flaw_lines,flaw_line_indices

@ray.remote
def read_file_label_batch(samples,tokenizer):
    batch_inputs,batch_labels=[],[]
    batch_flaw_lines, batch_flaw_indices = [], []
    for sample in samples:
        inputs,labels,flaw_lines,flaw_line_indices=read_file_label(sample,tokenizer)
        batch_inputs.extend(inputs)
        batch_labels.extend(labels)
        batch_flaw_lines.extend(flaw_lines)
        batch_flaw_indices.extend(flaw_line_indices)
    return batch_inputs,batch_labels,batch_flaw_lines,batch_flaw_indices



def prepare_dataset(samples,tokenizer):
        records=samples.to_dict("records")
        batch_size=3000
        chunk_size = 10  # 한 번에 10개 task만 처리
        source_codes, labels = [], []
        flaw_lines, flaw_line_indices = [], []
        
        all_batches = list(range(0, len(records), batch_size))
        
        for chunk_start in tqdm(range(0, len(all_batches), chunk_size)):
            process_examples = []
            chunk_end = min(chunk_start + chunk_size, len(all_batches))
            
            for i in all_batches[chunk_start:chunk_end]:
                process_examples.append(
                    read_file_label_batch.remote(records[i:i+batch_size], tokenizer)
                )
            
            # 10개씩만 join
            result = ray.get(process_examples)
            for source_code, label, f_line, f_index in result:
                source_codes.extend(source_code)
                labels.extend(label)
                flaw_lines.extend(f_line)
                flaw_line_indices.extend(f_index)
        
        return source_codes, labels, flaw_lines, flaw_line_indices



def train_filter(source_codes,labels,flaw_lines,flaw_line_indices):
    final_samples=defaultdict(dict)
    modified_source_codes,modified_labels=[],[]
    modified_flaw_lines, modified_flaw_indices = [], []

    for i,_ in tqdm(enumerate(labels),total=len(labels)):
            hash1=getMD5("".join(source_codes[i].split()))
            if hash1 not in final_samples:
                final_samples[hash1]["source_code"]=source_codes[i]
                final_samples[hash1]["label"]=labels[i]
                final_samples[hash1]["flaw_line"]=flaw_lines[i]
                final_samples[hash1]["flaw_line_index"]=flaw_line_indices[i]
            else:
                old_label=final_samples[hash1]["label"]
                if (old_label!=-1 and old_label!=labels[i]) or (old_label==-1):
                    final_samples[hash1]["label"]=-1

    
    for i in final_samples:
        if final_samples[i]["label"]!=-1:
            modified_source_codes.append(final_samples[i]["source_code"])
            modified_labels.append(final_samples[i]["label"])
            modified_flaw_lines.append(final_samples[i].get("flaw_line",""))
            modified_flaw_indices.append(final_samples[i].get("flaw_line_index",""))
            
    return modified_source_codes,modified_labels,modified_flaw_lines,modified_flaw_indices

def test_filter(source_codes,labels,flaw_lines,flaw_line_indices):
    collisions=0
    duplicates=0
    final_samples=defaultdict(dict)
    modified_source_codes,modified_labels=[],[]
    modified_flaw_lines, modified_flaw_indices = [], []

    for i,_ in tqdm(enumerate(labels),total=len(labels)):
            hash1=getMD5("".join(source_codes[i].split()))
            if hash1 not in final_samples:
                final_samples[hash1]["source_code"]=[source_codes[i]]
                final_samples[hash1]["label"]=[labels[i]]
                final_samples[hash1]["flaw_line"]=[flaw_lines[i]]
                final_samples[hash1]["flaw_line_index"]=[flaw_line_indices[i]]
            else:
                old_label=final_samples[hash1]["label"]
                if (old_label!=-1 and old_label!=labels[i]) or (old_label==-1):
                    final_samples[hash1]["label"]=-1
                    print(hash1)
                    collisions+=1
                else:
                    final_samples[hash1]["source_code"].append(source_codes[i])
                    final_samples[hash1]["label"].append(labels[i])
                    final_samples[hash1]["flaw_line"].append(flaw_lines[i])
                    final_samples[hash1]["flaw_line_index"].append(flaw_line_indices[i])
                    duplicates+=1
    
    for i in final_samples:
        if final_samples[i]["label"]!=-1:
            modified_source_codes.extend(final_samples[i]["source_code"])
            modified_labels.extend(final_samples[i]["label"])
            modified_flaw_lines.extend(final_samples[i]["flaw_line"])
            modified_flaw_indices.extend(final_samples[i]["flaw_line_index"])

    return modified_source_codes,modified_labels,modified_flaw_lines,modified_flaw_indices



def compute_metrics(p):    
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    confusion_matrix1 = confusion_matrix(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,"confusion_matrix":confusion_matrix1.tolist()
           }


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_csv_path", type=str, required=True,
                    help="The input training data file (a csv file).")
parser.add_argument("--tokenizer_name",  type=str, required=True,
                    help="tokenizer_name")
parser.add_argument("--model_name", type=str, required=True,
                    help="model path")
parser.add_argument("--output_dir", type=str, required=True,
                    help="output_dir")
parser.add_argument("--per_device_train_batch_size", type=int, required=True,
                    help="per_device_train_batch_size")
parser.add_argument("--per_device_eval_batch_size", type=int, required=True,
                    help="per_device_eval_batch_size")
parser.add_argument("--num_train_epochs", type=int, required=True,
                    help="num_train_epochs")
parser.add_argument("--dataset_path", type=str, required=True,
                    help="dataset_path")
parser.add_argument("--prepare_dataset", default=False,action='store_true',
                    help="prepare_dataset")
parser.add_argument("--train",default=False,action='store_true',
                    help="train")
parser.add_argument("--val_predict", default=False,action='store_true',
                    help="val_predict")
parser.add_argument("--test_predict", default=False,action='store_true',
                    help="test_predict")
parser.add_argument("--train_predict", default=False,action='store_true',
                    help="train_predict")
parser.add_argument("--data_type", type=str, choices=['train', 'val', 'test'], default=None,
                    help="If provided, treat the entire input file as this type (no splitting).")

args = parser.parse_args()

print("Arguments", args)

tokenizer_name=args.tokenizer_name
model_name=args.model_name
dataset_csv_path=args.dataset_csv_path
output_dir=args.output_dir
dataset_path= args.dataset_path

if not exists(dataset_path):
        os.makedirs(dataset_path)

if not exists(output_dir):
        os.makedirs(output_dir)
        



project_df=pd.read_csv(dataset_csv_path)

# Compatability for custom dataset schema
if "flaw_line_index" in project_df.columns and "vulnerable_line_numbers" not in project_df.columns:
    project_df["vulnerable_line_numbers"] = project_df["flaw_line_index"]

if "dataset_type" not in project_df.columns and args.data_type is None:
    print("Warning: 'dataset_type' column missing. Defaulting all data to 'train_val'.")
    project_df["dataset_type"] = "train_val"

project_df["vulnerable_line_numbers"]=project_df["vulnerable_line_numbers"].fillna("")


if args.prepare_dataset:
    print("Preparing Dataset...")
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    
    if args.data_type:
        # SINGLE DATASET MODE (No splitting)
        print(f"Processing entire file as SINGLE type: {args.data_type}")
        target_df = project_df # Use all rows
        
        raw_source, raw_labels = prepare_dataset(target_df, tokenizer)
        
        # Determine filter type (Train/Val get train_filter, Test gets test_filter)
        if args.data_type in ['train', 'val']:
            filtered_source, filtered_labels = train_filter(raw_source, raw_labels)
        else: # test
            filtered_source, filtered_labels = test_filter(raw_source, raw_labels)
            
        # Save CSV
        output_df = pd.DataFrame({
            "processed_func": filtered_source, 
            "target": filtered_labels
        })
        
        # Add metadata only if it's test set or if explicitly available
        if args.data_type == 'test':
             output_df["flaw_line"] = filtered_flaw_lines
             output_df["flaw_line_index"] = filtered_flaw_indices
             
        output_df.to_csv(join(dataset_path, f"{args.data_type}.csv"), index=False)
        print(f"Saved {args.data_type}.csv")
        # Tokenize
        if len(filtered_source) > 0:
            tokenized_input = tokenizer(filtered_source, padding=True, truncation=True, max_length=512)
            dataset_obj = Dataset(tokenized_input, filtered_labels)
        else:
            print(f"Warning: {args.data_type} dataset is empty after filtering.")
            dataset_obj = Dataset({"input_ids": []}, [])
            
        # Save PKL
        pkl_name = f"{args.data_type}_dataset.pkl"
        with open(join(dataset_path, pkl_name), "wb") as output_file:
            pickle.dump(dataset_obj, output_file)
            print(f"Saved {pkl_name}")

    else:
        # LEGACY MODE (Split based on 'dataset_type' or default split)
        train_val=project_df[project_df["dataset_type"]=="train_val"]
        test_data=project_df[project_df["dataset_type"]=="test"]

        # Train/Val processing
        source_code,labels,flaw_lines,flaw_indices=prepare_dataset(train_val,tokenizer)
        filtered_source_code,filtered_labels,filtered_flaw_lines,filtered_flaw_indices=train_filter(source_code,labels,flaw_lines,flaw_indices)
        
        # Split all 4 arrays
        train_source_code, val_source_code, train_labels, val_labels, \
        _, _, _, _ = train_test_split(
            filtered_source_code, filtered_labels, filtered_flaw_lines, filtered_flaw_indices, test_size=0.1
        )

        # Save train/val CSV (No metadata needed for training usually)
        pd.DataFrame({"processed_func": train_source_code, "target": train_labels}).to_csv(join(dataset_path, "train.csv"), index=False)
        pd.DataFrame({"processed_func": val_source_code, "target": val_labels}).to_csv(join(dataset_path, "val.csv"), index=False)

        X_chunked_train_tokenized = tokenizer(train_source_code,padding=True, truncation=True, max_length=512)
        X_chunked_val_tokenized = tokenizer(val_source_code,padding=True, truncation=True, max_length=512)
        train_dataset = Dataset(X_chunked_train_tokenized, train_labels)
        val_dataset = Dataset(X_chunked_val_tokenized, val_labels)
        with open(join(dataset_path,"train_dataset.pkl"), "wb") as output_file:
            pickle.dump(train_dataset, output_file)

        with open(join(dataset_path,"val_dataset.pkl"), "wb") as output_file:
            pickle.dump(val_dataset, output_file)

        # Test processing
        test_source_code,test_labels,test_flaw_lines,test_flaw_indices=prepare_dataset(test_data,tokenizer)
        filtered_test_source_code,filtered_test_labels,filtered_test_flaw_lines,filtered_test_flaw_indices=test_filter(test_source_code,test_labels,test_flaw_lines,test_flaw_indices)

        # Save test CSV with metadata
        pd.DataFrame({
            "processed_func": filtered_test_source_code, 
            "target": filtered_test_labels,
            "flaw_line": filtered_test_flaw_lines,
            "flaw_line_index": filtered_test_flaw_indices
        }).to_csv(join(dataset_path, "test.csv"), index=False)
        print("Saved train.csv, val.csv, test.csv")

        if len(filtered_test_source_code) > 0:
            X_chunked_test_tokenized = tokenizer(filtered_test_source_code,padding=True, truncation=True, max_length=512)
            test_dataset = Dataset(X_chunked_test_tokenized,filtered_test_labels) 
        else:
            print("Warning: Test dataset is empty after filtering. Skipping test dataset tokenization.")
            test_dataset = Dataset({"input_ids": []}, []) # Create dummy empty dataset

        with open(join(dataset_path,"test_dataset.pkl"), "wb") as output_file:
            pickle.dump(test_dataset, output_file)

else:
    print("Loading Dataset...")
    with open(join(dataset_path,"train_dataset.pkl"), "rb") as output_file_train:
        train_dataset = pickle.load(output_file_train)
    
    with open(join(dataset_path,"val_dataset.pkl"), "rb") as output_file_val:
        val_dataset= pickle.load(output_file_val)
    with open(join(dataset_path,"test_dataset.pkl"), "rb") as output_file_test:
        test_dataset= pickle.load(output_file_test)
    

if args.train:
    print("Training Dataset...")
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        seed=121,
        load_best_model_at_end=True,
        fp16=True)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    trainer.save_model(join(args.output_dir,"best_model"))

if args.val_predict or args.test_predict:
    best_model= RobertaForSequenceClassification.from_pretrained(join(args.output_dir,"best_model"), num_labels=2)
    train_args = TrainingArguments(output_dir=args.output_dir,per_device_eval_batch_size=args.per_device_eval_batch_size,fp16=True)
    trainer = Trainer(model=best_model,args=train_args)

if args.val_predict:
    print("Validation Results...")
    raw_pred_val, b, c = trainer.predict(val_dataset)
    y_pred_val = np.argmax(raw_pred_val, axis=1)
    val_metrics=compute_metrics([raw_pred_val,val_dataset.labels])
    print("Validation Metrics",val_metrics)
    
if args.test_predict:
    print("Test Results...")
    raw_pred_test, b, c = trainer.predict(test_dataset)
    y_pred_test = np.argmax(raw_pred_test, axis=1)
    test_preds=compute_metrics([raw_pred_test,test_dataset.labels])
    print("Test Metrics",test_preds)

    
if args.train_predict:
    print("Train Results...")
    raw_pred_train, b, c = trainer.predict(train_dataset)
    y_pred_train = np.argmax(raw_pred_train, axis=1)
    train_preds=compute_metrics([raw_pred_train,train_dataset.labels])
    print("Train Metrics",train_preds)
    log_file.write("Train Metrics:" +json.dumps(train_preds)+"\n")


