import torch
import time
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import functools

class BaseTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader,
                 device='cpu', validation_interval=0):
        ''' validation_interval is the number of batches to train on before
        validating and logging loss/acc to allow data points within each epoch '''
        self.model = model.to(device)
        self.criterion = criterion  #the loss function
        self.optimizer = optimizer  #the optimizer
        self.train_loader = train_loader  #the train loader
        self.val_loader = val_loader  #the valid loader
        self.device = device
        self.train_log = []
        self.val_log = []
        self.train_run = []
        # self.val_run = []

        self.validation_interval = validation_interval
        
    #the function to train the model in many epochs
    def fit(self, num_epochs):
        self.num_batches = len(self.train_loader)
        best_acc = 0.0
        best_train_acc = 0.0

        # Learning rate scheduler: Divide by 10 every 5 epochs
        # scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

        for epoch in range(num_epochs):
            start = time.time()
            print(f'Epoch {epoch + 1}/{num_epochs}')
            train_loss, train_accuracy, train_cm = self.train_one_epoch()
            val_loss, val_accuracy, val_cm = self.validate_one_epoch()
            # scheduler.step()
            end = time.time()
            # log results
            self.train_log.append((train_loss, train_accuracy))
            self.val_log.append((val_loss, val_accuracy))
            print(f"{self.num_batches}/{self.num_batches} - Time Taken: {(end-start)/60:.2f} - train_loss: {train_loss:.4f} - train_accuracy: {train_accuracy*100:.4f}% - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy*100:.4f}%")

            # save model based on validation result
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train': (train_loss, train_accuracy),
                'val': (val_loss, val_accuracy),
                'train_cm': train_cm,
                'val_cm': val_cm,
                'model':self.model,
                }, 's3d_rgb_best.pth')

            elif (val_accuracy==best_acc) and (train_accuracy >=best_train_acc):
                best_acc = val_accuracy
                best_train_acc = train_accuracy
                torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train': (train_loss, train_accuracy),
                'val': (val_loss, val_accuracy),
                'train_cm': train_cm,
                'val_cm': val_cm,
                'model':self.model,
                }, 's3d_rgb_best.pth')


            # save latest model
            torch.save({
            'epoch': epoch+1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train': self.train_log,
            'val': self.val_log,
            'train_run':self.train_run,
            # 'val_run':self.val_run,
            'train_cm': train_cm,
            'val_cm': val_cm,
            'model':self.model,
            }, 's3d_rgb_last.pth') 

        return self.train_log, self.val_log

    #train in one epoch, return the train_acc, train_loss
    def train_one_epoch(self):
        self.model.train()
        device = self.device
        running_loss, correct, total = 0.0, 0, 0
        train_run = []
        # val_run = []
        all_preds = []
        all_labels = []

        start = time.time()

        for i, data in enumerate(self.train_loader):
            # print(f"training: {i}")
            # start = time.time()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()

            if self.device != 'cpu':
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(inputs)
                    # print(outputs.shape)
                    # print(outputs)
                    # print(torch.softmax(outputs, dim=1))
                    # print()
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # print("metrics")
            # end = time.time()
            # print(f"Time Taken: {end-start}")

            # every validation_interval batches, log loss, acc
            if (self.validation_interval > 0) and ((i+1) % self.validation_interval == 0):
                acc = correct / total
                los = running_loss / (i+1)
                train_run.append((los,acc))

                # v_loss, v_acc = self.validate_one_epoch()
                # val_run.append((v_loss, v_acc))
                # self.model.train()

                end = time.time()
                print(f"{i+1}/{self.num_batches} - Time Taken: {(end-start)/60:.2f} - train_loss: {los:.4f} - train_accuracy: {acc*100:.4f}%") # - val_loss: {v_loss:.4f} - val_accuracy: {v_acc*100:.4f}%
                start = time.time()
                # break

        train_accuracy = correct / total
        train_loss = running_loss / self.num_batches

        self.train_run.append(train_run)
        # self.val_run.append(val_run)

        cm = confusion_matrix(all_labels, all_preds)

        return train_loss, train_accuracy, cm

    #evaluate on a loader and return the loss and accuracy
    def evaluate(self, loader):
        self.model.eval()
        device = self.device
        loss, total_loss, correct, total = 0.0, 0.0, 0, 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i, data in enumerate(loader):
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

                # Collect predictions and labels for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                # print(all_preds)
                # print(all_labels)

                # every validation_interval batches, log loss, acc
                # if (self.validation_interval > 0) and ((i+1) % self.validation_interval == 0):
                #     acc = correct / total
                #     los = total_loss / (i+1)

                #     end = time.time()
                #     print(f"{i+1}/{self.num_batches} - Time Taken: {(end-start)/60:.2f} - train_loss: {los:.4f} - train_accuracy: {acc*100:.4f}%") # - val_loss: {v_loss:.4f} - val_accuracy: {v_acc*100:.4f}%
                #     start = time.time()
                    # break
        

        accuracy = correct / total
        loss = total_loss / len(loader)
        # print(total_loss,len(loader))

        # Generate the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        return loss, accuracy, cm

    #return the val_acc, val_loss, be called at the end of each epoch
    def validate_one_epoch(self):
        val_loss, val_accuracy, val_cm = self.evaluate(self.val_loader)
        return val_loss, val_accuracy, val_cm
    
    def validate_train_loader(self):
        print('Evaluating Train_loader....')
        train_loss, train_accuracy, train_cm = self.evaluate(self.train_loader)
        return train_loss, train_accuracy, train_cm
    
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

def infer(model, loader, criterion, device: str):
    ''' Get outputs, loss, accuracy, confusion matrix of inference '''
    model.eval()
    device = device
    loss, total_loss, correct, total = 0.0, 0.0, 0, 0
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if device != 'cpu':
                with torch.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            all_outputs.extend(torch.softmax(outputs, dim=1))
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    loss = total_loss / len(loader)

    # Generate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    return all_outputs, all_labels, loss, accuracy, cm
    
def ensemble(model1, model2, loader1, loader2, criterion, device: str):
    ''' return tuple with two items. raw outputs of models, and averaged
    combined outputs, for ensemble inferencing. '''

    outputs1, all_labels1, loss1, acc1, conf_matrix1 = infer(model1, loader1, criterion, device)
    outputs2, all_labels2, loss2, acc2, conf_matrix2 = infer(model2, loader2, criterion, device)

    if all_labels1 == all_labels2:
        labels = torch.tensor(all_labels1).to(torch.long).to(device)

    combined_output = (torch.stack(outputs1)*0.5) + (torch.stack(outputs2)*0.5)
    
    _, predicted = torch.max(combined_output, 1)

    loss = criterion(combined_output.to(torch.float32), labels)
    acc = (predicted == labels).sum().item() / len(labels)

    cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())

    return loss, acc, cm