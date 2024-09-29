import torch
import time
import torch.optim as optim

class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.criterion = criterion  #the loss function
        self.optimizer = optimizer  #the optimizer
        self.train_loader = train_loader  #the train loader
        self.val_loader = val_loader  #the valid loader
        self.device = device
        self.train_log = []
        self.val_log = []

    #the function to train the model in many epochs
    def fit(self, num_epochs):
        self.num_batches = len(self.train_loader)
        best_acc = 0.0

        # Learning rate scheduler: Divide by 10 every 5 epochs
        # scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        for epoch in range(num_epochs):
            start = time.time()
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate_one_epoch()
            # scheduler.step()
            end = time.time()
            # log results
            self.train_log.append((train_loss, train_accuracy))
            self.val_log.append((val_loss, val_accuracy))
            print(f"{self.num_batches}/{self.num_batches} - Time Taken: {(end-start)/60} - train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy*100:.4f}% - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy*100:.4f}%")

            # save model based on validation result
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train': (train_loss, train_accuracy),
                'val': (val_loss, val_accuracy),
                }, 's3d_rgb_best.pth') 

            # save latest model
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train': self.train_log,
            'val': self.val_log,
            }, 's3d_rgb_last.pth') 

        
        return self.train_log, self.val_log

    #train in one epoch, return the train_acc, train_loss
    def train_one_epoch(self):
        self.model.train()
        device = self.device
        running_loss, correct, total = 0.0, 0, 0

        for i, data in enumerate(self.train_loader):
            # print(f"training: {i}")
            # start = time.time()

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()

            if self.device != 'cpu':
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(inputs)
                    # print(f"output")
                    loss = self.criterion(outputs, labels)
                    # print(f"loss")
                    loss.backward()
                    # print(f"backward")
                    self.optimizer.step()
                    # print("step")
            else:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print("metrics")
            # end = time.time()
            # print(f"Time Taken: {end-start}")
        
        train_accuracy = correct / total
        train_loss = running_loss / self.num_batches

        return train_loss, train_accuracy

    #evaluate on a loader and return the loss and accuracy
    def evaluate(self, loader):
        self.model.eval()
        device = self.device
        loss, total_loss, correct, total = 0.0, 0.0, 0, 0

        with torch.no_grad():
            for data in loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                if self.device != 'cpu':
                    with torch.autocast(device_type="cuda"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                else:
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = total_loss / len(loader)
        return loss, accuracy

    #return the val_acc, val_loss, be called at the end of each epoch
    def validate_one_epoch(self):
        val_loss, val_accuracy = self.evaluate(self.val_loader)
        return val_loss, val_accuracy
    
def freeze(model: torch.nn.Module):
    ''' Disable model training '''
    _set_freeze(model, freeze=True)

def unfreeze(model: torch.nn.Module):
    ''' Enable model training '''
    _set_freeze(model, freeze=False)

def _set_freeze(model: torch.nn.Module, freeze: bool):
    ''' Private helper function. Sets model freeze state. '''
    for param in model.parameters():
        param.requires_grad = not freeze