import torch.nn as nn
from transformers import Trainer, TrainingArguments
import torch

class Multitask_Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract multiple labels from inputs
        labels_task1 = inputs.pop("task1_labels")
        labels_task2 = inputs.pop("task2_labels")

        # Forward pass
        outputs = model(**inputs)
        # Assuming your model returns logits for both tasks in the outputs
        logits_task1 = outputs.get("logits_task1") 
        logits_task2 = outputs.get("logits_task2")

        # Compute custom loss for each task
        loss_fct_task1 = nn.CrossEntropyLoss()
        loss_fct_task2 = nn.BCEWithLogitsLoss() # Example loss

        loss1 = loss_fct_task1(logits_task1.view(-1, self.model.config.num_labels_task1), labels_task1.view(-1))
        loss2 = loss_fct_task2(logits_task2.view(-1, self.model.config.num_labels_task2), labels_task2.float().view(-1, self.model.config.num_labels_task2))

        # Combine losses
        total_loss = loss1 + loss2 

        return (total_loss, outputs) if return_outputs else total_loss

# Use the custom trainer
training_args = TrainingArguments(
    output_dir="./mtl_results",
    label_names=["task1_labels", "task2_labels"], # Inform Trainer of all label columns
    # ... other arguments
)

trainer = CustomMtlTrainer(
    model=your_custom_mtl_model,
    args=training_args,
    train_dataset=your_mtl_train_dataset,
    # ...
)