from layers import Classifier_DNN, Concentration
import torch.nn as nn


class PCpredict(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_class: int,
            use_bn: bool = False,
            drop_rate: float = 0.3,
            pooling_type: str = 'mean',
    ) -> None:
        super().__init__()
        self.dropout = drop_rate
        self.concentration = Concentration(in_channels, pooling_type=pooling_type)
        self.classfier = Classifier_DNN(in_channels, num_class)

    def forward(self, x1, GP_info):
        Z = self.concentration(x1, GP_info)
        out = self.classfier(Z)
        return out

