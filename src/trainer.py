import wandb
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AdamW
from utils.constant import PROJECT_ROOT
from utils.determine_device import determine_device

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, args):
        self.device = determine_device()
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.args = args
        self.best_val_loss = float("inf")
        self.save_path = Path(PROJECT_ROOT) / self.args.output


    def train_one_epoch(self, epoch):
        """
        Function to train and validate the model for one epoch, log the losses to wandb, 
        and save the model if the validation loss improves.

        Args:
        - epoch: The current epoch number.

        Returns:
        - best_val_loss: The best validation loss encountered so far.
        """
        self.model.train()  # Set model to training mode
        train_loss = 0.0

        # ---- Training Phase ----
        for batch in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}"):
            # Move data to the same device as the model
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Calculate loss
            loss = outputs.loss
            train_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log training loss per batch
            wandb.log({"train_loss_batch": loss.item(), "epoch": epoch + 1})

        # Log average train loss for the epoch
        avg_train_loss = train_loss / len(self.train_dataloader)
        wandb.log({"train_loss_epoch": avg_train_loss, "epoch": epoch + 1})
        print(f"Training loss for epoch {epoch + 1}: {avg_train_loss}")

        # ---- Validation Phase ----
        self.model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for batch in tqdm(self.val_dataloader, desc=f"Validation epoch {epoch+1}"):
                # Move data to the same device as the model
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

                # Calculate loss
                loss = outputs.loss
                val_loss += loss.item()

        # Log validation loss for the epoch
        avg_val_loss = val_loss / len(self.val_dataloader)
        wandb.log({"val_loss_epoch": avg_val_loss, "epoch": epoch + 1})
        print(f"Validation loss for epoch {epoch + 1}: {avg_val_loss}")

        # Checkpoint saving: Save if the current validation loss is better
        if avg_val_loss < self.best_val_loss:
            print(f"Validation loss improved, saving model checkpoint!")
            self.best_val_loss = avg_val_loss
            # Save model checkpoint
            self.model.save_pretrained(self.save_path)
            self.optimizer.save_pretrained(self.save_path)

    def train(self):
        """
        Main training loop that calls `train_one_epoch` for each epoch.
        """
        # Initialize WandB
        wandb.init(project=self.args.wandb_project, config=self.args, name=f"training_run")

        for epoch in range(self.args.epochs):
            print(f"Starting epoch {epoch + 1}")
            self.train_one_epoch(epoch)
        
        print("Training complete!")
        wandb.finish()