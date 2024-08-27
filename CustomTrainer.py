import torch
import math
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler, default_data_collator, AutoModelForMaskedLM
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class CustomTrainer:

    def __init__(self, model, tokenizer, device, train_dataset, eval_dataset, data_collator, batch_size=64, num_train_epochs=5):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.batch_size = batch_size
        self.num_train_epochs = num_train_epochs
        self.train_losses = []
        self.eval_losses = []
        self.accuracies = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []

    def train(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=self.batch_size, collate_fn=default_data_collator
        )

        optimizer = AdamW(self.model.parameters(), lr=5e-5)

        accelerator = Accelerator()

        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader
        )

        num_train_epochs = self.num_train_epochs
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        progress_bar = tqdm(range(num_training_steps))

        # Pre-Training Evaluation
        self.evaluate(eval_dataloader, model, accelerator, epoch="Pre-Training")

        for epoch in range(num_train_epochs):
            # Training
            model.train()
            for batch in train_dataloader:
                outputs = model(**batch)
                loss = outputs.loss
                self.train_losses.append(loss.item())
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            # Evaluation
            self.evaluate(eval_dataloader, model, accelerator, epoch=epoch)

    def evaluate(self, eval_dataloader, model, accelerator, epoch=None):
        model.eval()
        losses = []
        all_preds = []
        all_labels = []

        for step, batch in enumerate(eval_dataloader):
            with torch.inference_mode():
                outputs = model(**batch)

            loss = outputs.loss
            self.eval_losses.append(loss.item())
            losses.append(accelerator.gather(loss.repeat(self.batch_size)))

            # Gather predictions and true labels for masked tokens
            predictions = outputs.logits.argmax(dim=-1)
            mask_token_index = batch['input_ids'] == self.tokenizer.mask_token_id
            masked_preds = torch.masked_select(predictions, mask_token_index)
            masked_labels = torch.masked_select(batch['labels'], mask_token_index)

            all_preds.extend(accelerator.gather(masked_preds).cpu().numpy())
            all_labels.extend(accelerator.gather(masked_labels).cpu().numpy())

        # Loss & Perplexity
        losses = torch.cat(losses)
        losses = losses[: len(self.eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        self.accuracies.append(accuracy)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)

        print(f">>> {epoch}: Perplexity: {perplexity:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
