import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class Classifier(nn.Module):
    classifier_type = "fc"

    def __init__(
        self,
        features_dim,
        n_classes,
        init="kaiming"
    ):
        super().__init__()

        self.features_dim = features_dim
        self.n_classes = n_classes
        self.init_method = init


    @property
    def weights(self):
        return torch.cat([w for w in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    @property
    def bias(self):
        if self._bias is not None:
            return torch.cat([b for b in self._bias])
        return None

    @property
    def new_bias(self):
        return self._bias[-1]

    @property
    def old_bias(self):
        if len(self._bias) > 1:
            return self._bias[:-1]
        return None

    def forward(self, features):
        if len(self._weights) == 0:
            raise Exception("Add some classes before training.")

        weights = self.weights
        if self._negative_weights is not None and (
            self.training is True or self.eval_negative_weights
        ) and self.use_neg_weights:
            weights = torch.cat((weights, self._negative_weights), 0)

        if self.normalize:
            features = F.normalize(features, dim=1, p=2)

        logits = F.linear(features, weights, bias=self.bias)
        return {"logits": logits}

    def add_classes(self, n_classes):
        self._weights.append(nn.Parameter(torch.randn(n_classes, self.features_dim)))
        self._init(self.init_method, self.new_weights)

        if self.use_bias:
            self._bias.append(nn.Parameter(torch.randn(n_classes)))
            self._init(0., self.new_bias)

        self.to(self.device)

    def reset_weights(self):
        self._init(self.init_method, self.weights)

    @staticmethod
    def _init(init_method, parameters):
        if isinstance(init_method, float) or isinstance(init_method, int):
            nn.init.constant_(parameters, init_method)
        elif init_method == "kaiming":
            nn.init.kaiming_normal_(parameters, nonlinearity="linear")
        else:
            raise NotImplementedError("Unknown initialization method: {}.".format(init_method))

    def align_weights(self):
        """Align new weights based on old weights norm.

        # Reference:
            * Maintaining Discrimination and Fairness in Class Incremental Learning
              Zhao et al. 2019
        """
        with torch.no_grad():
            old_weights = torch.cat([w for w in self.old_weights])

            old_norm = torch.mean(old_weights.norm(dim=1))
            new_norm = torch.mean(self.new_weights.norm(dim=1))

            self._weights[-1] = nn.Parameter((old_norm / new_norm) * self._weights[-1])

    def align_features(self, features):
        avg_weights_norm = self.weights.data.norm(dim=1).mean()
        avg_features_norm = features.data.norm(dim=1).mean()

        features.data = features.data * (avg_weights_norm / avg_features_norm)
        return features

    def add_custom_weights(self, weights, ponderate=None, **kwargs):
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                weights = weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_new_weights_norm = weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_new_weights_norm
                weights = weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        self._weights.append(nn.Parameter(weights))
        self.to(self.device)
