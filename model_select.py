import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import Models

sys.path.append("..")


class ModelCalled(nn.Module):

    def __init__(self, arch, num_classes=9):
        super(ModelCalled, self).__init__()
        self.model = Models.__dict__[arch](num_classes=num_classes)

    def forward(self, x):
        out = self.model.forward(x)
        return out


class Trainer(object):
    def __init__(self, name, num_classes=9):
        super(Trainer, self).__init__()
        self.model = ModelCalled(name, num_classes)
        assert isinstance(name, str)

        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, weight_decay=1e-3)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer,
                                                                 milestones=[5, 15, 30],
                                                                 gamma=0.1)

    def init(self):
        self.model = self.model.to(self.device)

    def fit(self, train_loader, epochs, val_loader=None):
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            self.model.train()

            print("\nEpoch {}/{}".format(epoch + 1, epochs))

            with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
                for step, batch in enumerate(train_loader):
                    X_batch = batch['spectrogram'].to(self.device)
                    y_batch = batch['label'].to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(True):
                        # forward + backward
                        outputs = self.model(X_batch)
                        batch_loss = self.criterion(outputs, y_batch)
                        batch_loss.backward()

                        # update the parameters
                        self.optimizer.step()

                    pbar.update(1)
                self.lr_scheduler.step()    # 执行学习率更新

            # model evaluation - train data
            train_loss, train_acc = self.evaluate(train_loader)
            print("loss: %.4f - accuracy: %.4f" % (train_loss, train_acc), end='')

            # model evaluation - validation data
            val_loss, val_acc = None, None
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                print(" - val_loss: %.4f - val_accuracy: %.4f" % (val_loss, val_acc))

            # store the model's training progress
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

        return history

    def predict(self, X):
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X)

        return outputs

    def evaluate(self, data_loader):
        running_loss = torch.tensor(0.0).to(self.device)
        running_acc = torch.tensor(0.0).to(self.device)

        batch_size = torch.tensor(data_loader.batch_size).to(self.device)

        for step, batch in enumerate(data_loader):
            X_batch = batch['spectrogram'].to(self.device)
            y_batch = batch['label'].to(self.device)

            outputs = self.predict(X_batch)

            # get batch loss
            loss = self.criterion(outputs, y_batch)
            running_loss = running_loss + loss

            # calculate batch accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions = (predictions == y_batch).float().sum()
            running_acc = running_acc + torch.div(correct_predictions, batch_size)

        loss = running_loss.item() / (step + 1)
        accuracy = running_acc.item() / (step + 1)

        return loss, accuracy


if __name__ == "__main__":
    net = Trainer("vgg11")

    print(net.model)
