from typing import Dict, Text

import shap
from shap.maskers import Text as TextMasker

from . import BaseExplainer
from .explanation import Explanation
from .utils import parse_explainer_args


class SHAPExplainer(BaseExplainer):
    NAME = "Partition SHAP"

    def compute_feature_importance(self, text, target=1, **explainer_args):
        init_args, call_args = parse_explainer_args(explainer_args)

        # SHAP silent mode
        init_args["silent"] = init_args.get("silent", True)
        # Default to 'Partition' algorithm
        # init_args["algorithm"] = init_args.get("algorithm", "permutation")
        init_args["algorithm"] = init_args.get("algorithm", "partition")
        #  seed for reproducibility
        init_args["seed"] = init_args.get("seed", 42)

        def func(texts):
            _, logits = self.helper._forward(texts)
            if logits.size(-1) == 1:
                return logits.cpu().numpy()
            return logits.softmax(-1).cpu().numpy()

        masker = TextMasker(self.tokenizer)
        explainer_partition = shap.Explainer(model=func, masker=masker, **init_args)

        if explainer_args["baseline"] is not None:
            call_args["baseline"] = explainer_args["baseline"]
        if init_args["algorithm"] == "permutation":
            call_args["max_evals"] = "auto"

        shap_values = explainer_partition([text], **call_args)
        attr = shap_values.values[0][:, target]
        base_values = shap_values.base_values

        output = Explanation(text, self.get_tokens(text), attr, self.NAME, target, base_values)
        return output
