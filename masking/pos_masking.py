import collections
import numpy as np
from transformers import default_data_collator


class pos_colator:
     
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    

    #def whole_word_random_data_collator(features):
    def __call__(self, features):
        mlm_probability = 0.15
        pos_to_mask = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "WRB", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        for feature in features:
            word_ids = feature.pop("word_ids")
            word_ids_decoded = self.tokenizer.decode(feature["input_ids"])
            pos_tags = self.model(word_ids_decoded)
            #print(pos_tags)
            adjective_indices = [pos['index'] - 1 for pos in pos_tags if pos['entity'] in pos_to_mask]
            #print(adjective_pos)
            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1

            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    current_word_index += 1
                    mapping[current_word_index].append(idx)

            # Randomly mask words
            #mask = np.random.binomial(1, mlm_probability, (len(mapping),))
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            new_labels = [-100] * len(labels)

            #for word_id in np.where(mask)[0]:
            for word_id in adjective_indices:
                #word_id = word_id.item()
                for idx in mapping[word_id]:
                    new_labels[idx] = labels[idx]
                    input_ids[idx] = self.tokenizer.mask_token_id
            feature["labels"] = new_labels

        return default_data_collator(features)