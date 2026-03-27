# finetune.py
import os
import wandb


import logging
from datasets import load_dataset
from pathlib import Path

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
)

from utils import ( 
    compute_metrics, 
    load_model_params,
    load_data_paths,
    preprocess_dataset,
    cleanup_trainer_memory,
    save_best_sweep_results
)

from cvae import XLMRobertaCVAE, CVAEDataCollator
from models import (
    CustomModel, 
    MultiTaskCascadeCustomModel, 
    CustomClassifierOutput, 
    MultiTaskCustomModel,
    MultiTaskDataCollator
    ) 

import torch

# Fixed paths 
DATA_DIR = Path("data/")
MODELS_DIR = Path("models/")
SWEEP_DIR = Path('sweeps/')

os.environ["WANDB_PROJECT"] = "BEA26-Task" 
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_WATCH"]="false" # turn off watch to log faster

pos_list = ['noun',
 'adverb',
 'adjective',
 'verb',
 'preposition',
 'misc',
 'number',
 'not-no',
 'determiner'
 ]

def run_finetune(model_params_path, models_to_run, seed, sweep=False, n_trials=5):
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

    set_seed(seed)
    
    # Loop over models defined in the parameter CSV
    for row in load_model_params(model_params_path, models_to_run):
        
        model_name = row["model_name"]
        l1 = row["L1"]
        is_baseline = "baseline" in model_name

        num_heads_raw = row.get('num_heads')
        num_heads = int(num_heads_raw) if num_heads_raw else None
        
        last_k_layer_raw = row.get('last_k_layer')
        last_k_layer = int(last_k_layer_raw[-1]) if last_k_layer_raw else None
        
        mtl_row = row.get('mtl')
        mtl = True if int(mtl_row) == 1 else False
        
        pred_head = row.get('pred_head').lower() if row.get('pred_head') else None
        layer_pool = row.get('layer_pool').lower() if row.get('layer_pool') else None
        token_pool = row.get('token_pool').lower() if row.get('token_pool') else None
        num_pos_labels = row.get('num_pos_labels') if row.get('num_pos_labels') else len(pos_list)
        # Fallback base values if Optuna isn't modifying them
        base_dropout = float(row.get('dropout', 0.1))
        base_lr = float(row.get("learning_rate", 2e-5))
        base_wd = float(row.get("weight_decay", 0.1))
        base_warmup = int(row.get("warmup_steps", 100))
        epochs = int(row.get("epochs", 5))
        batch_size = int(row.get("batch_size", 64))
        
        config = AutoConfig.from_pretrained(row["pretrained_model"], num_labels=1)
        
        # CRITICAL: manually pass custom arguments into the config, such that it
        # would be saved along with the model.
        config.update({
            'pred_head':pred_head, 
            'token_pool':token_pool,
            'layer_pool':layer_pool,
            'dropout':base_dropout,
            'last_k_layer':last_k_layer,
            'num_heads':num_heads
        })
        
        if mtl:
            config.update({
                'num_pos_labels': len(pos_list)
            })
        
        logging.info(f"\n{'='*50}\nProcessing Model: {model_name}\n{'='*50}")
        
        if "baseline" in model_name:
            logging.info(f"Baseline model: {model_name}")
        else:
            logging.info(f"""
                ---------------------Model Setup---------------------
                Using regression head: {pred_head}
                Layer-wise aggregation: {layer_pool if layer_pool else False}
                Token-wise aggregation: {token_pool if token_pool else False}
                Last k layer: {last_k_layer if last_k_layer else None}
                Multi-task Training?: {mtl}
                -----------------------------------------------------""")

            # if pred_head == 'vib':
            #     assert beta is not None and z_dim is not None, \
            #         f"Model {model_name} is VIB but missing 'annealing_factor' or 'latent_dim' in parameters."        
            #     logging.info(f"VIB Config -> Beta: {beta}, Latent Dim: {z_dim}")

                
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
            
            # Itemize POS
            pos_to_idx = {pos:i for i, pos in enumerate(pos_list)}
            tokenized_ds = tokenized_ds.map(
                lambda x: {"pos_labels": [pos_to_idx[val] for val in x['en_target_pos']]},
                batched=True,
                remove_columns=['en_target_pos'],
                desc="Itemizing POS tagging"
            )
            
            def model_init_function(trial=None):
                if is_baseline:
                    return AutoModelForSequenceClassification.from_pretrained(
                        row["pretrained_model"], config=config
                    )
                elif mtl:
                    return MultiTaskCascadeCustomModel.from_pretrained(
                        row['pretrained_model'],
                        config=config
                    )
                else: 
                    return CustomModel.from_pretrained(
                        row["pretrained_model"], 
                        config=config  
                    )
            
            # Dictionary to hold the winning params (empty if no sweep)
            best_hp = {}        
    
            if sweep:
                # --- The Sweep Trainer ---
                logging.info(f"Starting {int(n_trials)}-trial hyperparameter sweep for {model_name}...")
                
                def optuna_hp_space(trial):
                    return {
                        "learning_rate": trial.suggest_categorical("learning_rate", [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]),
                        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 0.1]),
                        "warmup_steps": trial.suggest_categorical('warmup_steps', [100, 200])
                    }
    
                sweep_args = TrainingArguments(
                    eval_strategy="epoch",
                    save_strategy="no",          
                    load_best_model_at_end=False,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    metric_for_best_model="eval_rmse",
                    greater_is_better=False, 
                    # report_to="wandb",
                )
                
                if mtl:
                    sweep_args.label_names = ["labels", "pos_labels"]
                    data_collator = MultiTaskDataCollator(tokenizer)
                else:
                    data_collator = DataCollatorWithPadding(tokenizer)

                sweep_trainer = Trainer(
                    model=None, 
                    model_init=model_init_function,
                    args=sweep_args,
                    train_dataset=tokenized_ds["train"],
                    eval_dataset=tokenized_ds["validation"],
                    data_collator=data_collator,
                    compute_metrics=compute_metrics
                )
                
                best_run = sweep_trainer.hyperparameter_search(
                    direction="minimize", 
                    compute_objective=lambda metrics:metrics['eval_rmse'],
                    backend="optuna",
                    n_trials=n_trials,
                    hp_space=optuna_hp_space,
                    hp_name=lambda trial: f"{model_name}_trial_{trial.number}"
                )
                
                best_hp = best_run.hyperparameters
                wandb.finish() # Closes the sweep logging securely
                
                logging.info(f"Sweep complete for {model_name}! Best RMSE: {best_run.objective}")
                logging.info(f"Best Hyperparameters: {best_run.hyperparameters}")
                
                # save sweep results
                save_best_sweep_results(model_name, best_run, SWEEP_DIR)
                logging.info(f"Sweep results save to {SWEEP_DIR}")

            # --- The Final "Golden" Training Run ---
            logging.info("Initializing ultimate training run...")
            
            # Combine sweep results with base defaults
            final_lr = best_hp.get("learning_rate", base_lr)
            final_wd = best_hp.get("weight_decay", base_wd)
            final_warmup = best_hp.get("warmup_steps", base_warmup)
            
            final_args = TrainingArguments(
                output_dir=str(MODELS_DIR / model_name),
                run_name=f"{model_name}",
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                save_total_limit=1,
                metric_for_best_model="eval_rmse",     
                greater_is_better=False,          
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=final_lr,
                weight_decay=final_wd,
                warmup_steps=final_warmup,
                # report_to="wandb",
                seed=seed
            )
        
            if mtl:
                final_args.label_names = ["labels", "pos_labels"]
                data_collator = MultiTaskDataCollator(tokenizer)
            else:
                data_collator = DataCollatorWithPadding(tokenizer)
            
            # Manually instantiate the winning Architecture
            if is_baseline:
                final_model = AutoModelForSequenceClassification.from_pretrained(
                    row["pretrained_model"], config=config
                )
            elif mtl:
                final_model = MultiTaskCascadeCustomModel.from_pretrained(
                        row['pretrained_model'],
                        config=config
                    )
            else:
                final_model = CustomModel.from_pretrained(
                    row["pretrained_model"],
                    config=config, 
                )
            
            final_trainer = Trainer(
                model=final_model,
                args=final_args,
                train_dataset=tokenized_ds["train"],
                eval_dataset=tokenized_ds["validation"],
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )

            # Train and Serialize the final weights
            final_trainer.train()
            wandb.finish()
            
            final_trainer.save_model(MODELS_DIR / model_name)
            tokenizer.save_pretrained(MODELS_DIR / model_name)
            logging.info(f"Final model saved at {str(MODELS_DIR / model_name)}")
            
            # Free memory after training
            cleanup_trainer_memory(final_trainer, tokenized_ds, preprocessed_ds)
            if sweep:
                cleanup_trainer_memory(sweep_trainer)
        
        except Exception:
            logging.exception(f"Failed model {model_name}")
            raise