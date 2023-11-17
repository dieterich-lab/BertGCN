import math
import pdb
from typing import List, Union

import numpy as np
import torch
from tqdm.autonotebook import tqdm
from transformers.tokenization_utils_base import BatchEncoding


class ModelHelper:
    """
    Wrapper class to interface with HuggingFace models
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _tokenize(self, text: str, **tok_kwargs) -> BatchEncoding:
        """
        Base tokenization strategy for a single text.

        Note that we truncate to the maximum length supported by the model.

        :param text str: the string to tokenize
        """
        return self.tokenizer(text, return_tensors="pt", truncation=True, **tok_kwargs)

    def get_input_embeds(self, text: str) -> torch.Tensor:
        """Extract input embeddings

        :param text str: the string to extract embeddings from.
        """
        item = self._tokenize(text)
        item = {k: v.to(self.model.device) for k, v in item.items()}
        embeddings = self._get_input_embeds_from_ids(item["input_ids"][0])
        embeddings = embeddings.unsqueeze(0)
        return embeddings

    def _get_input_embeds_from_ids(self, ids) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def get_tokens(self, text: str, **tok_kwargs) -> List[str]:
        """Extract a list of tokens

        :param text str: the string to extract tokens from.
        """
        item = self._tokenize(text)
        input_len = item["attention_mask"].sum()
        ids = item["input_ids"][0][:input_len]
        return self.tokenizer.convert_ids_to_tokens(ids, **tok_kwargs)

    def _forward_with_input_embeds(
        self,
        input_embeds,
        attention_mask=None,
        batch_size=8,
        show_progress=False,
        output_hidden_states=False,
    ):
        input_len = input_embeds.shape[0]
        n_batches = math.ceil(input_len / batch_size)
        input_batches = torch.tensor_split(input_embeds, n_batches)
        # mask_batches = torch.tensor_split(attention_mask, n_batches)

        if show_progress:
            pbar = tqdm(total=n_batches, desc="Batch", leave=False)

        outputs = list()
        for emb in input_batches:
        # for emb, mask in zip(input_batches, mask_batches):
            out = self.model(
                emb
                # inputs_embeds=emb,
                # attention_mask=mask,
                # output_hidden_states=output_hidden_states,
            )
            outputs.append(out)

            if show_progress:
                pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat(outputs)
        # logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    def _forward(
        self,
        embeds: Union[str, List[str]],
        batch_size=8,
        show_progress=False,
        use_input_embeddings=False,
        output_hidden_states=True,
        **tok_kwargs
    ):
        # if isinstance(embeds, str):
        #     embeds = [embeds]

        device = None
        n_batches = math.ceil(len(embeds) / batch_size)
        if type(embeds) == torch.Tensor:
            batches = np.array_split(embeds, n_batches)
        else:
            # because for GCN the input is huge (node_ids x node_ids)
            device = embeds[0].device
            n = np.array([x.detach().cpu().squeeze().numpy() for x in embeds])
            batches = np.array_split(n, n_batches)
            # batches = [torch.tensor(x, device=device) for x in batches]
            # batches = torch.tensor_split(torch.cat([x.detach() for x in embeds]), n_batches)

        outputs = list()
        with torch.no_grad():
            if show_progress:
                pbar = tqdm(total=n_batches, desc="Batch", leave=False)

            for batch in batches:
                # if isinstance(embeds[0], str):
                #     item = self._tokenize(batch.tolist(), padding="longest", **tok_kwargs)
                # elif isinstance(embeds[0], list):
                #     item = {"input_ids": torch.tensor(batch)}
                # item = {k: v.to(self.model.device) for k, v in item.items()}

                # if use_input_embeddings:
                    # ids = item.pop("input_ids")  # (B,S,d_model)
                    # input_embeddings = self._get_input_embeds_from_ids(ids)
                if device is not None:
                    out = self.model(
                        torch.tensor(batch, device=device),
                    )
                else:
                    out = self.model(
                        torch.tensor(batch)
                    )
                # else:
                #     out = self.model(**item, output_hidden_states=output_hidden_states)
                outputs.append(out)

                if show_progress:
                    pbar.update(1)

        if show_progress:
            pbar.close()

        logits = torch.cat(outputs)
        # logits = torch.cat([o.logits for o in outputs])
        return outputs, logits

    # def _get_class_predicted_probability(self, text, tokenizer, target):
    #     outputs = self._forward(text, tokenizer)
    #     logits = outputs.logits[0]
    #     class_prob = logits.softmax(-1)[target].item()
    #     return class_prob

    # def _get_tokenizer(self, tokenizer=None):
    #     tokenizer = tokenizer if tokenizer else self.tokenizer
    #     if tokenizer is None:
    #         raise ValueError("Tokenizer is not specified")
    #     return tokenizer

    # def get_predicted_label(self, text, tokenizer=None):
    #     tokenizer = self._get_tokenizer(tokenizer)
    #     outputs = self._forward(text, tokenizer)
    #     logits = outputs.logits

    #     prediction = logits.argmax(-1).item()
    #     return prediction

    # # TODO - Uniformate
    # def _get_class_predicted_probabilities_texts(self, texts, tokenizer, target):
    #     # TODO
    #     tokenizer = tokenizer if tokenizer else self.tokenizer
    #     if tokenizer is None:
    #         raise ValueError("Tokenizer is not specified")
    #     inputs = tokenizer(texts, return_tensors="pt", padding="longest")

    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     return outputs.logits.softmax(-1)[:, target]

    # def _forward(self, idx, tokenizer=None, no_grad=True, use_inputs=False):
    #     self.model.eval()
    #     tokenizer = tokenizer if tokenizer else self.tokenizer
    #     if tokenizer is None:
    #         raise ValueError("Tokenizer is not specified")

    #     item = tokenizer(idx, return_tensors="pt")

    #     def _foward_pass(use_inputs=False):

    #         if use_inputs:
    #             embeddings = self._get_input_embeds(idx)
    #             outputs = self.model(
    #                 inputs_embeds=embeddings,
    #                 **item,
    #                 output_hidden_states=True,
    #             )

    #             return outputs, embeddings

    #         else:
    #             outputs = self.model(
    #                 **item, output_attentions=True, output_hidden_states=True
    #             )
    #             return outputs

    #     if no_grad:
    #         with torch.no_grad():
    #             outputs = _foward_pass(use_inputs)
    #     else:
    #         outputs = _foward_pass(use_inputs)

    #     return outputs
