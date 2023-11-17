import warnings

import numpy as np
import torch
from ignite.metrics import Metric

# These decorators helps with distributed settings
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from sklearn.metrics import classification_report


class SklearnClassificationReport(Metric):
    def __init__(
        self,
        target_names=None,
        arznei=False,
        output_transform=lambda x: x,
        device="cpu",
    ):
        if arznei:
            self.arznei = list()
        self.target_names = target_names
        self.y_pred, self.y = list(), list()
        super(SklearnClassificationReport, self).__init__(
            output_transform=output_transform, device=device
        )

    @reinit__is_reduced
    def reset(self):
        self.y_pred, self.y = list(), list()
        super(SklearnClassificationReport, self).reset()

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output[0].detach(), output[1].detach()

        y_pred = torch.argmax(y_pred, dim=1)

        y_pred = y_pred.cpu().tolist()
        y = y.cpu().tolist()

        self.y.extend(y)
        self.y_pred.extend(y_pred)

        if type(self.arznei) == list:
            arznei = output[2].detach().cpu().tolist()
            self.arznei.extend(arznei)

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if type(self.arznei) == list:
            a_ypred, a_y = list(), list()
            for i, (x, y) in enumerate(zip(self.y_pred, self.y)):
                a_y.append(self.arznei[i])
                if x == y:
                    a_ypred.append(self.arznei[i])
                else:
                    a_ypred.append(10000)
            target_names = [self.target_names[i] for i in np.unique(a_y)]
            warnings.filterwarnings("ignore")
            report = classification_report(
                a_y,
                a_ypred,
                target_names=target_names,
                labels=np.unique(a_y),
                zero_division=0,
            )
            warnings.resetwarnings()
        else:
            target_names = [self.target_names[i] for i in np.unique(self.y)]
            warnings.filterwarnings("ignore")
            report = classification_report(
                self.y,
                self.y_pred,
                target_names=target_names,
                labels=np.unique(self.y),
                zero_division=0,
            )
            warnings.resetwarnings()
        return report