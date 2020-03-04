import torch
import torch.nn as nn
import torch.nn.init as init

dropout_prob = 0.5


class FlatCnnLayer(nn.Module):
    def __init__(self, embedding_size, sequence_length, filter_sizes=[3, 4, 5], out_channels=128):
        super(FlatCnnLayer, self).__init__()

        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.out_channels = out_channels

        self.filter_layers = nn.ModuleList()
        for filter_size in filter_sizes:
            self.filter_layers.append(self._make_filter_layer(filter_size))
        self.dropout = nn.Dropout(p=dropout_prob)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal(m.weight, mean=0, std=0.1)
                init.constant(m.bias, 0.1)

    def forward(self, x):
        pools = []
        for filter_layer in self.filter_layers:
            pools.append(filter_layer(x))
        x = torch.cat(pools, dim=1)

        x = x.view(x.size()[0], -1)
        x = self.dropout(x)

        return x

    def _make_filter_layer(self, filter_size):
        return nn.Sequential(
            nn.Conv2d(1, self.out_channels,
                      (filter_size, self.embedding_size)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((self.sequence_length - filter_size + 1, 1), stride=1)
        )
