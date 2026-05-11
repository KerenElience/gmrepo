import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer():
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                 optimizer: torch.optim, criterion, num_epochs: int, l1: float, early_stop: int, device: torch.device):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.l1 = l1
        self.early_stop = early_stop

        self.model = model.to(device)
        self.device = device

        self.losses = {"train": [], "test": []}
    
    def train(self):
        train_loss = 0.0
        self.model.train()
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device).view(-1, 1)
            self.optimizer.zero_grad()
            outs = self.model(x)
            loss = self.criterion(outs, y)
            if self.l1 > 0:
                l1_reg = 0.0
                for param in self.model.parameters():
                    l1_reg += param.abs().sum().item()
                loss += self.l1 * l1_reg
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss/len(self.train_loader)   

    def eval(self):
        test_loss = 0.0
        correct = 0
        total = 0
        all_prob = []
        all_true = []
        self.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                y = y.to(self.device).view(-1, 1)
                outs = self.model(x)
                loss = self.criterion(outs,y)
                test_loss += loss.item()
                predicted = (outs > 0.5).float()
                correct += (predicted == y.float()).sum().item()
                total += y.shape[0]
                all_prob.append(torch.sigmoid(outs.cpu()))
                all_true.append(y.cpu())
        probs = torch.cat(all_prob).numpy().flatten()
        y_true = torch.cat(all_true).numpy().flatten()
        return test_loss/len(self.test_loader), correct, total, probs, y_true
    
    def save(self, savepath):
        torch.save(self.model.state_dict(), savepath)

    def run(self, savepath: str = "disease_bineary_classifiy_best.pth"):
        no_improve = 0
        best_loss = 1.0
        best_probs = None
        best_true = None

        for epoch in range(self.num_epochs):
            train_loss = self.train()
            test_loss, correct, total, prob, y_true = self.eval()
            # if epoch % 20 == 0:
            print(f"Epoch [{epoch+1}/{self.num_epochs}]:"\
                  f"Train loss: {train_loss}, Test loss: {test_loss}, "\
                  f"accuary: {correct/total*100:.2f}%[{correct}/{total}]")
            self.losses["train"].append(train_loss)
            self.losses["test"].append(test_loss)
            if test_loss < best_loss:
                no_improve = 0
                best_loss = test_loss
                best_probs = prob
                best_true = y_true
                self.save(savepath)
            else:
                no_improve += 1
            if no_improve > self.early_stop:
                print(f"Epoch [{epoch+1}/{self.num_epochs}]: "\
                      f"Train loss: {train_loss}, Test loss: {test_loss}, "\
                      f"accuary: {correct/total*100:.2f}%[{correct}/{total}]")
                print(f"best_loss: {best_loss}")
                break
        return best_probs, best_true