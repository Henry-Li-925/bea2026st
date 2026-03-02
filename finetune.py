# finetune.py
import logging
from datasets import load_dataset
from pathlib import Path

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    set_seed,
)

from utils import ( 
    compute_metrics, 
    load_model_params,
    load_data_paths,
    preprocess_dataset,
    cleanup_trainer_memory,
    custom_optimizer
)

from cvae import XLMRobertaCVAE, CVAEDataCollator
from models import CloseTrack_Predictor
import torch

# Fixed paths 
DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")

def run_finetune(model_params_path, models_to_run, seed):
    """
    Fine-tune pre-trained transformer models based on a CSV of parameters.

    Args:
        model_params_path (Path): CSV file with model parameters and metadata.
        models_to_run (List[str]): List of models to finetune.
        seed (int, optional): Random seed.
    """
    
    # Checking cuda availability
    if torch.cuda.is_available():
        logging.info(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("Training on CPU.")

    # Loop over models defined in the parameter CSV
    for row in load_model_params(model_params_path, models_to_run):
        
        model_name = row["model_name"]
        l1 = row["L1"]

        beta = row.get('annealing_factor')
        z_dim = row.get('latent_dim')
        dropout = row.get("dropout")
        pred_head = row.get('pred_head')
        layer_wise = row.get('layer_wise')
        token_wise = row.get('token_wise')
        pos_encoding = row.get('pos_encoding')
        learned_pos = row.get('learned_pos')
        max_seq_len = row.get('max_seq_len')
        # Only cast if they exist, otherwise leave as None
        if beta is not None:
            beta = float(beta)
        if z_dim is not None:
            z_dim = int(z_dim)
        if dropout is not None:
            dropout = float(dropout)
        if max_seq_len is not None:
            max_seq_len = int(max_seq_len)
        
        if "baseline" in model_name:
            logging.info(f"Baseline model: {model_name}")
            
            def model_init_function():
                return AutoModelForSequenceClassification.from_pretrained(
                    row["pretrained_model"], num_labels=1
                )
        
        else:
            logging.info(f"""
                ---------------------Model Setup---------------------
                Using regression head: {pred_head}
                Layer-wise aggregation: {layer_wise if layer_wise else False}
                Token-wise aggregation: {token_wise if token_wise else False}
                Positional Encoding: {bool(int(pos_encoding)) if pos_encoding else False}
                Learned Positional Encoder: {bool(int(learned_pos)) if learned_pos else False}
                Dropout: {dropout}
                -----------------------------------------------------""")

            if pred_head == 'vib':
                assert beta is not None and z_dim is not None, \
                    f"Model {model_name} is VIB but missing 'annealing_factor' or 'latent_dim' in parameters."        
                logging.info(f"VIB Config -> Beta: {beta}, Latent Dim: {z_dim}")

            if pos_encoding:
                logging.info(f"Maximum Sequence Length for Positional Encoder: {max_seq_len}")
                
            def model_init_function():
                return CloseTrack_Predictor.from_pretrained(
                    row['pretrained_model'], 
                    num_labels=1,
                    dropout=dropout,
                    latent_dim=z_dim,
                    beta=beta,
                    layer_wise=layer_wise,
                    token_wise=token_wise,
                    pred_head=pred_head,
                    pos_encoding=pos_encoding,
                    learned_pos=learned_pos,
                    max_seq_len=max_seq_len
                )

        try:
            logging.info(f"Fine-tuning model: {model_name}...")

            # Load dataset paths and Hugging Face DatasetDict
            data_files = load_data_paths(DATA_DIR, l1, "finetune")
            hf_dataset = load_dataset("csv", data_files=data_files)
    
            # Load tokenizer and prepare input text formatting
            tokenizer = AutoTokenizer.from_pretrained(row["pretrained_model"], use_fast=True)
            cols_to_merge = row["component_order"].split("; ")
            sep_token = f" {tokenizer.sep_token} " if tokenizer.sep_token else " "
    
            # Preprocess dataset: format input text, rename target label and remove extra columns
            preprocessed_ds = preprocess_dataset(hf_dataset, cols_to_merge, sep_token)
            
            # Tokenize dataset
            tokenized_ds = preprocessed_ds.map(
                lambda x: tokenizer(x["input_text"], truncation=True),
                batched=True,
                desc="Tokenizing input text"
            )

            # Itemize L1 
            l1_to_idx = {l1:i for i, l1 in enumerate(["es", "de", "cn"])}
            tokenized_ds = tokenized_ds.map(
                lambda x: {"l1_encode": [l1_to_idx[val] for val in x['L1']]},
                batched=True,
                remove_columns=['L1'],
                desc="Itemizing L1"
            )

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(MODELS_DIR / model_name),
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                save_total_limit=1,
                num_train_epochs=int(row["epochs"]),
                per_device_train_batch_size=int(row["batch_size"]),
                per_device_eval_batch_size=int(row["batch_size"]),
                learning_rate=float(row["learning_rate"]),
                weight_decay=float(row["weight_decay"]),
                warmup_ratio=float(row["warmup_ratio"]),
                load_best_model_at_end=True,
                report_to="none",
                seed=seed,
            )

            # Initialise trainer
            if "cvae" in model_name.lower():
                data_collator = CVAEDataCollator(tokenizer, custom_features=["l1_labels", "labels"])
            else:
                data_collator = DataCollatorWithPadding(tokenizer)

            # set customized optimizer
            # model = model_init_function()
            # optimizer = custom_optimizer(model, training_args.learning_rate, training_args.weight_decay, acc_lr = 1e-3) 
            trainer = Trainer(
                model_init=model_init_function,
                args=training_args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["validation"],
                data_collator=data_collator,
                # optimizers=(optimizer, None), # customizer optimizer
                compute_metrics=compute_metrics
            )

            # Verify model is on the correct device
            logging.info(f"Model successfully loaded on: {trainer.model.device}")

            # Finetune model and save
            trainer.train()
            trainer.save_model(MODELS_DIR / model_name)
            logging.info(f"Model {model_name} fine-tuned and saved at {str(MODELS_DIR / model_name)}")
            
            # Free memory after training
            cleanup_trainer_memory(trainer, tokenized_ds, preprocessed_ds)
        
        except Exception:
            logging.exception(f"Failed model {model_name}")
            raise