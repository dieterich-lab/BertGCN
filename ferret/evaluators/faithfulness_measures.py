import copy
from typing import List

import numpy as np
import torch
from scipy.stats import kendalltau

from ..explainers.explanation import Explanation, ExplanationWithRationale
from ..model_utils import ModelHelper
from . import BaseEvaluator
from .evaluation import Evaluation
from .utils_from_soft_to_discrete import (
    _check_and_define_get_id_discrete_rationale_function,
    parse_evaluator_args,
)


def _compute_aopc(scores):
    from statistics import mean

    return mean(scores)


class AOPC_Comprehensiveness_Evaluation(BaseEvaluator):
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def compute_evaluation(self, explanation: Explanation, target=1, **evaluation_args) -> Evaluation:
        """Evaluate an explanation on the AOPC Comprehensiveness metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args.
                We currently support multiple approaches to define the hard rationale from
                soft score rationales, based on:
                - th : token greater than a threshold
                - perc : more than x% of the tokens
                - k: top k values

        Returns:
            Evaluation : the AOPC Comprehensiveness score of the explanation
        """

        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        embeds = explanation.embeds
        score_explanation = explanation.scores
        _, logits = self.helper._forward(embeds=embeds, output_hidden_states=False)
        baseline = logits.softmax(-1)[0, target].item()

        # TODO - use tokens
        # Tokenized sentence
        # item = self.helper._tokenize(text)

        # Get token ids of the sentence
        # input_len = embeds.size(1)
        # input_ids = torch.arange(input_len)
        # input_ids = item["input_ids"][0][:input_len].tolist()

        # If remove_first_last, first and last token id (CLS and ) are removed
        # if remove_first_last == True:
        #     input_ids = input_ids[1:-1]
        #     if self.tokenizer.cls_token == explanation.tokens[0]:
        #         score_explanation = score_explanation[1:-1]

        discrete_expl_ths = list()
        id_tops = list()

        get_discrete_rationale_function = _check_and_define_get_id_discrete_rationale_function(
            removal_args["based_on"]
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:
            # Get rationale from score explanation
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

            # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
            if id_top is not None and last_id_top is not None and set(id_top) == last_id_top:
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Comprehensiveness
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.

            # For the comprehensiveness: we remove the terms in the discrete rationale.
            sample = torch.clone(embeds)

            sample[:, id_top, :] = sample[:, id_top, :].zero_()

            # discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids)

            discrete_expl_ths.append(sample)
            # discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == []:
            return Evaluation(self.SHORT_NAME, 0)

        # Prediction probability for the target
        _, logits = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        # if len(logits.size()) == 2:
        #     probs_removing = logits[:, target].cpu().numpy()
        # else:
        #     probs_removing = logits.softmax(-1)[:, target].cpu().numpy()
        probs_removing = logits.softmax(-1)[:, target].cpu().numpy()

        # compute probability difference
        removal_importance = baseline - probs_removing
        #  compute AOPC comprehensiveness
        aopc_comprehesiveness = _compute_aopc(removal_importance)

        evaluation_output = Evaluation(self.SHORT_NAME, aopc_comprehesiveness)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)


class AOPC_Sufficiency_Evaluation(BaseEvaluator):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    # Lower is better
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "faithfulness"

    def compute_evaluation(self, explanation: Explanation, target=1, **evaluation_args) -> Evaluation:
        """Evaluate an explanation on the AOPC Sufficiency metric.

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the AOPC Sufficiency score of the explanation
        """

        remove_first_last, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        embeds = explanation.embeds
        score_explanation = explanation.scores

        _, logits = self.helper._forward(embeds, output_hidden_states=False)
        baseline = logits.softmax(-1)[0, target].item()

        # TO DO - use tokens

        # Tokenized sentence
        # input_len = embeds.size(1)
        # input_ids = torch.arange(input_len)

        # If remove_first_last, first and last token id (CLS and ) are removed
        # if remove_first_last == True:
        #     input_ids = input_ids[1:-1]
        #     if self.tokenizer.cls_token == explanation.tokens[0]:
        #         score_explanation = score_explanation[1:-1]

        discrete_expl_ths = []
        id_tops = []

        get_discrete_rationale_function = _check_and_define_get_id_discrete_rationale_function(
            removal_args["based_on"]
        )

        thresholds = removal_args["thresholds"]
        last_id_top = None
        for v in thresholds:
            # Get rationale
            id_top = get_discrete_rationale_function(score_explanation, v, only_pos)

            # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
            if id_top is not None and last_id_top is not None and set(id_top) == last_id_top:
                id_top = None

            id_tops.append(id_top)

            if id_top is None:
                continue

            last_id_top = set(id_top)

            # Sufficiency
            # The only difference between comprehesivenss and sufficiency is the computation of the removal.

            # For the sufficiency: we keep only the terms in the discrete rationale.

            # sample = torch.clone(embeds)
            # sample = np.array(copy.copy(input_ids))

            # We take the tokens in the original order
            id_top = np.sort(id_top)

            sample = torch.zeros_like(embeds)
            sample[:, id_top, :] = embeds[:, id_top, :]

            # discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids)

            discrete_expl_ths.append(sample)
            # discrete_expl_ths.append(discrete_expl_th)

        if discrete_expl_ths == []:
            return Evaluation(self.SHORT_NAME, 1)

        # Prediction probability for the target
        _, logits = self.helper._forward(discrete_expl_ths, output_hidden_states=False)
        # if len(logits.size()) == 2:
        #     probs_removing = logits[:, target].cpu().numpy()
        # else:
        #     probs_removing = logits.softmax(-1)[:, target].cpu().numpy()
        probs_removing = logits.softmax(-1)[:, target].cpu().numpy()

        # Compute probability difference
        removal_importance = baseline - probs_removing
        aopc_sufficiency = _compute_aopc(removal_importance)

        evaluation_output = Evaluation(self.SHORT_NAME, aopc_sufficiency)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)


class TauLOO_Evaluation(BaseEvaluator):
    NAME = "tau_leave-one-out_correlation"
    SHORT_NAME = "taucorr_loo"
    TYPE_METRIC = "faithfulness"
    BEST_SORTING_ASCENDING = False

    def compute_leave_one_out_occlusion(
        self, embeds, target=1, remove_first_last=True, handle_tokens="remove", transform=None
    ):
        _, logits = self.helper._forward(embeds, output_hidden_states=False)
        if len(logits.size()) == 2:
            pred = logits[:, target].cpu().numpy()
        else:
            pred = logits.softmax(-1)[:, target].cpu().numpy()

        # item = self.helper._tokenize(embeds)
        input_len = embeds.size(1)
        # input_len = item["attention_mask"].sum().item()
        tokens = list(map(str, range(input_len)))
        # input_ids = item["input_ids"][0][:input_len].tolist()
        # if remove_first_last == True:
        #     input_ids = input_ids[1:-1]

        samples = list()
        # tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        for occ_idx in range(embeds.size(1)):
            # for occ_idx in range(len(input_ids)):
            sample = torch.clone(embeds)
            # sample = copy.copy(embeds)
            # if handle_tokens == "remove":
            #     sample.pop(occ_idx)
            # else:
            #     sample[occ_idx] = self.tokenizer.mask_token_id
            sample[:, occ_idx, :].zero_()
            # sample = self.tokenizer.decode(sample)
            # sample = [self.tokenizer.cls_token_id] + sample + [self.tokenizer.sep_token_id]
            samples.append(sample)

        _, logits = self.helper._forward(samples, output_hidden_states=False)
        if len(logits.size()) == 2:
            leave_one_out_removal = logits[:, target].cpu()
        else:
            leave_one_out_removal = logits.softmax(-1)[:, target].cpu()

        if transform is None:
            occlusion_importance = leave_one_out_removal - pred
        else:
            occlusion_importance = transform(leave_one_out_removal) - transform(pred)

        return occlusion_importance
        # return occlusion_importance, pred, tokens

    def compute_evaluation(self, explanation: Explanation, target=1, **evaluation_args) -> Evaluation:
        """Evaluate an explanation on the tau-LOO metric,
        i.e., the Kendall tau correlation between the explanation scores and leave one out (LOO) scores,
        computed by leaving one feature out and computing the change in the prediciton probability

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the tau-LOO score of the explanation
        """

        remove_first_last = evaluation_args.get("remove_first_last", True)

        embeds = explanation.embeds
        score_explanation = explanation.scores

        if remove_first_last:
            if self.tokenizer.cls_token == explanation.tokens[0]:
                score_explanation = score_explanation[1:-1]

        loo_scores = (
            self.compute_leave_one_out_occlusion(embeds, target=target, remove_first_last=remove_first_last).numpy() * -1
        )

        kendalltau_score = kendalltau(loo_scores, score_explanation)[0]

        evaluation_output = Evaluation(self.SHORT_NAME, kendalltau_score)
        return evaluation_output

    # def aggregate_score(self, score, total, **aggregation_args):
    #     return super().aggregate_score(score, total, **aggregation_args)
