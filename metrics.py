from ignite.metrics import Metric
from sklearn.metrics import (
	classification_report,
)
import torch

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class SklearnClassificationReport(Metric):
	def __init__(self, target_names=None, labels=None, output_transform=lambda x: x, device="cpu"):
		self.target_names = target_names
		self.labels = labels
		self.y_pred, self.y = list(), list()
		super(SklearnClassificationReport, self).__init__(output_transform=output_transform, device=device)

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

	@sync_all_reduce("_num_examples", "_num_correct:SUM")
	def compute(self):
		report = classification_report(
			self.y, self.y_pred, target_names=self.target_names, labels=self.labels, zero_division=0
		)
		return report
