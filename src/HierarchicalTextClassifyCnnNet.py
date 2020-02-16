import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from FlatCnnLayer import FlatCnnLayer
from TreeTools import TreeTools
import multiprocessing
import numpy as np

batch_size = 128
n_epochs = 200
display_step = 5
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)


class HierarchicalTextClassifyCnnNet(nn.Module):
    def __init__(self, embedding_size, sequence_length, tree, filter_sizes=[3, 4, 5], out_channels=128):
        super(HierarchicalTextClassifyCnnNet, self).__init__()

        self._tree_tools = TreeTools()
        self.tree = tree
        # create a weight matrix and bias vector for each node in the tree
        self.fc = nn.ModuleList([nn.Linear(out_channels * len(filter_sizes), len(subtree[1])) for subtree in
                                 self._tree_tools.get_subtrees(tree)])

        self.value_to_path_and_nodes_dict = {}
        for path, value in self._tree_tools.get_paths(tree):
            nodes = self._tree_tools.get_nodes(tree, path)
            self.value_to_path_and_nodes_dict[value] = path, nodes

        self.flat_layer = FlatCnnLayer(embedding_size, sequence_length, filter_sizes=filter_sizes,
                                       out_channels=out_channels)

        self.features = nn.Sequential(self.flat_layer)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight, gain=np.sqrt(2.0))
                init.constant(m.bias, 0.1)

    def forward(self, inputs, targets):
        features = self.features(inputs)
        predicts = map(self._get_predicts, features, targets)
        losses = map(self._get_loss, predicts, targets)
        return losses, predicts

    def _get_loss(self, predicts, label):
        path, _ = self.value_to_path_and_nodes_dict[int(label.data[0])]
        criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available:
            criterion = criterion.cuda()

        def f(predict, p):
            p = torch.LongTensor([p])
            # convert to cuda tensors if cuda flag is true
            if torch.cuda.is_available:
                p = p.cuda()
            p = Variable(p)
            return criterion(predict.unsqueeze(0), p)

        loss = map(f, predicts, path)
        return torch.sum(torch.cat(loss))

    def _get_predicts(self, feature, label):
        _, nodes = self.value_to_path_and_nodes_dict[int(label.data[0])]

        predicts = map(lambda n: self.fc[n](feature), nodes)
        return predicts


def fit(model, data, save_path):
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model, criterion = model.cuda(), criterion.cuda()

    # for param in list(model.parameters()):
    #     print(type(param.data), param.size())

    # optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x_train, x_test = torch.from_numpy(data['X_train']).float(), torch.from_numpy(data['X_test']).float()
    y_train, y_test = torch.from_numpy(data['Y_train']).int(), torch.from_numpy(data['Y_test']).int()

    train_set = TensorDataset(x_train, y_train)
    test_set = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=N_WORKERS,
                              pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=N_WORKERS)

    model.train()

    for epoch in range(1, n_epochs + 1):  # loop over the dataset multiple times

        acc_loss = 0.0

        for inputs, labels in iter(train_loader):
            # convert to cuda tensors if cuda flag is true
            if torch.cuda.is_available:
                inputs, labels = inputs.cuda(), labels.cuda()
            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            losses, _ = model(inputs, labels)
            loss = torch.mean(torch.cat(losses, dim=0))
            acc_loss += loss.data[0]

            loss.backward()
            optimizer.step()

        # print statistics
        if epoch % display_step == 0 or epoch == 1:
            print('[%3d] loss: %.5f' %
                  (epoch, acc_loss / len(train_set.data_tensor)))

    print('\rFinished Training\n')

    model.eval()

    nb_test_corrects, nb_test_samples = 0, 0

    for inputs, labels in iter(test_loader):
        # convert to cuda tensors if cuda flag is true
        if torch.cuda.is_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # forward + backward + optimize
        _, predicts = model(inputs, labels)

        nb_test_samples += labels.size(0)
        for predicted, label in zip(predicts, labels):
            nb_test_corrects += _check_predicts(model, predicted, label)

    print ('Accuracy of the network {:.2f}% ({:d} / {:d})'.format(
        100 * nb_test_corrects / nb_test_samples,
        nb_test_corrects,
        nb_test_samples)
    )

    torch.save(model.flat_layer.state_dict(), save_path)


def _check_predicts(model, predicts, label):
    path, _ = model.value_to_path_and_nodes_dict[int(label.data[0])]
    for predict, p in zip(predicts, path):
        if np.argmax(predict.data) != p:
            return 0
    return 1
