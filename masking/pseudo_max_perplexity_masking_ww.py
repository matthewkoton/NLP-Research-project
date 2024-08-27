import collections
import torch
import numpy as np
from transformers import default_data_collator


class psudo_max_loss_data_collator_ww:
     
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    

    #:def pseudo_max_perplexity_data_collator(features, n=5):
    def __call__(self, features, n=10):
        pmp_probability = 0.15 # probability of a word being chosen to mask

        for feature in features:
            word_ids = feature.pop("word_ids")
            # Create a map between non-special tokens and corresponding real token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Variables to track the best masking configuration
            best_perplexity = float('-inf')
            best_masked_input_ids = None
            best_labels = None

            for _ in range(n):
                # Randomly mask words
                mask = np.random.binomial(1, pmp_probability, (len(mapping),))
                input_ids = torch.tensor(feature["input_ids"]).clone().detach().to(self.device)  # Add batch dimension
                labels = input_ids.clone()  # Clone to avoid modifying the original
                new_labels = [-100] * len(labels)
                for word_id in np.where(mask)[0]:
                    word_id = word_id.item()
                    for idx in mapping[word_id]:
                        new_labels[idx] = labels[idx]
                        input_ids[idx] = self.tokenizer.mask_token_id

                # Calculate perplexity for this masked input
                with torch.inference_mode():
                    outputs = self.model(input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
                    loss = outputs.loss
                    perplexity = torch.exp(loss).item()

                # Track the best masking based on perplexity
                if perplexity > best_perplexity:
                    best_perplexity = perplexity
                    best_masked_input_ids = input_ids
                    best_labels = new_labels

            # Update feature with the best masking configuration
            feature["input_ids"] = best_masked_input_ids
            feature["labels"] = best_labels

        return default_data_collator(features)
