'''
engine.py -> contains the train_step(), test_step() , and traind_model() functions.
'''
import torch 
from torch import nn
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device 
):
    """
    trains the pytorch model
    forward pass  -> loss calculation -> backward pass -> optimizer step
    args:
        model:torch.nn.model - the model we are goint to train
        dataloader : torch.utils.data.DataLoader - the dataloader to use for training
        loss_fn: torch.nn.Module - the loss function to use for training
        optimizer: torch.optim.Optimizer - the optimizer to use for training
        device: torch.device - the device to use for training (cpu or cuda)
    """
    model.train()
    train_loss , train_acc =0,0
    for batch ,(X,y) in enumerate(dataloader):
        # now first we need to send our batch data to the device
        X, y  = X.to(device), y.to(device)
        #1.forward pass
        y_pred = model(X)
        #@2 clauclate the laoss
        loss = loss_fn(y_pred, y)# remeber this is the loss at this bartch step
        train_loss += loss.item() # add the loss to the total loss
        #3 optimizer zero grad 
        optimizer.zero_grad()
        #4 backward pass
        loss.backward()
        #5 optimizer step
        optimizer.step()
        #6 calculate the accuracy
        y_pred_class = torch.argmax(y_pred, dim=1)
        acc = torch.sum(y_pred_class == y).item() / len(y_pred_class)
        train_acc += acc
        if batch % 100 == 0:
            print(f"Batch: {batch} | Loss: {loss.item():.5f} | Acc: {acc:.2f}")
    # calculate the average loss and accuracy
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    print(f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}")
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    tests the pytorch model
    forward pass -> loss calculation
    args:
        model:torch.nn.model - the model we are goint to test
        dataloader : torch.utils.data.DataLoader - the dataloader to use for testing
        loss_fn: torch.nn.Module - the loss function to use for testing
        device: torch.device - the device to use for testing (cpu or cuda)
    """
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # send data to device
            X, y = X.to(device), y.to(device)
            # forward pass
            y_pred = model(X)
            # calculate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()        
            # calculate accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            acc = torch.sum(y_pred_class == y).item() / len(y_pred_class)
            test_acc += acc
            if batch % 100 == 0:
                print(f"Batch: {batch} | Loss: {loss.item():.5f} | Acc: {acc:.2f}")
    # calculate the average loss and accuracy
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    print(f"Test Loss: {test_loss:.5f} | Test Acc: {test_acc:.2f}")
    return test_loss, test_acc
def train_model(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                device: torch.device) -> Dict[str, List[float]]:
    """ trains the pytorch model
        passes the target to the pytorch moidel throught the train_step() nd test_step() functions
        """
    results ={
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        print(f"Epoch: {epoch+1}/{epochs}")
        # train the model
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        # test the model
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        # append the results to the results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        print("-" * 50)
    print("Training complete!")
    return results

