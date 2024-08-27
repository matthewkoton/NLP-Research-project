import collections
import torch
import math
from transformers import default_data_collator

class min_loss_data_collator:
     
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
          
    def __call__(self, features):
            wwm_probability = 0.15

            for feature in features:
                word_ids = feature.pop("word_ids")

                # Create a map between words and corresponding token indices
                mapping = collections.defaultdict(list)
                current_word_index = -1
                for idx, word_id in enumerate(word_ids):
                    if word_id is not None:
                        current_word_index += 1
                        mapping[current_word_index].append(idx)

                input_ids = torch.tensor(feature["input_ids"]).clone().detach().to(self.device)  # Add batch dimension
                labels = input_ids.clone()  # Clone to avoid modifying the original

                with torch.inference_mode():
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    logits = self.model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0)).logits
                    # Reshape logits and labels to the appropriate shape for loss calculation
                    logits = logits.view(-1, logits.size(-1))  # Flatten logits
                    labels = labels.view(-1)  # Flatten labels

                    losses = loss_fct(
                        logits.view(-1, logits.size(-1)),  # Flatten logits
                        labels.view(-1)                    # Flatten labels
                    )
                    n = math.ceil(losses.size(0) * wwm_probability)
                    # Reshape the losses to match the original batch and sequence lengths
                    per_token_loss = losses.view(-1)
                    top_loss_indices = torch.topk(per_token_loss[1:-1], n, largest=False).indices + 1

                # Mask the tokens with the highest loss
                new_labels = [-100] * len(feature["labels"])
                for idx in top_loss_indices:
                    idx = idx.item()
                    for token_idx in mapping[idx]:
                        new_labels[token_idx] = labels[token_idx].item()
                        input_ids[token_idx] = self.tokenizer.mask_token_id

                feature["input_ids"] = input_ids.tolist()
                feature["labels"] = new_labels

            return default_data_collator(features)
    
