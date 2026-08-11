from utils import utils
from tqdm.auto import tqdm
import torch
from torch import nn
import numpy as np

# setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"



def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion_instance,
               criterion_cluster,
               optimizer: torch.optim.Optimizer,
               scheduler,
               weight,
               device=device,
               run_visual_only=False):
    """
    适配双流输入 (Dual Visual + Dual Text) 的训练步骤
    """
    model.train()
    train_loss = 0


    for batch, (img1, img2, input_ids_weak, mask_weak, input_ids_strong, mask_strong) in enumerate(dataloader):
        
 
        img1 = img1.to(device)
        img2 = img2.to(device)
        
        input_ids_weak = input_ids_weak.to(device)
        mask_weak = mask_weak.to(device)
        
        input_ids_strong = input_ids_strong.to(device)
        mask_strong = mask_strong.to(device)

        # 3. Forward pass
        z_i, z_j, c_i, c_j = model(
            img1, img2, 
            input_ids_weak, mask_weak,   
            input_ids_strong, mask_strong,
            run_visual_only=run_visual_only 
        )

        # 4. Calculate the loss
        loss_instance = criterion_instance(z_i, z_j)
        loss_cluster = criterion_cluster(c_i, c_j)
        loss = weight * loss_instance + (1 - weight) * loss_cluster
        train_loss += loss.item()

        # 5. Backward & Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if scheduler is not None:
        scheduler.step()

    train_loss = train_loss / len(dataloader)
    return train_loss


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
              device=device,
              run_visual_only=False): 

    model.eval()
    test_loss = 0
    
 
    all_preds = []
    all_targets = []

    with torch.inference_mode():
        for batch, (img, input_ids, attention_mask, y) in enumerate(dataloader):
            img = img.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)

            test_pred_logits = model.forward_cluster(
                img, 
                input_ids, 
                attention_mask, 
                run_visual_only=run_visual_only
            )


            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
    
            all_preds.append(test_pred_labels.cpu().numpy())
            all_targets.append(y.cpu().numpy())

 
    test_loss = test_loss / len(dataloader)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    ind, test_acc = utils.cluster_acc(all_targets, all_preds)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler,
          weight,
          criterion_instance,
          criterion_cluster,
          epochs: int,
          device=device,
          run_visual_only=False):
  
    results = {"train_loss": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs), desc="Training"):
        if scheduler:
             lr = scheduler.get_last_lr()[0]
        else:
             lr = optimizer.param_groups[0]["lr"]

 
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                criterion_instance=criterion_instance,
                                criterion_cluster=criterion_cluster,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                weight=weight,
                                device=device,
                                run_visual_only=run_visual_only) 

        if test_dataloader:
            test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            device=device,
                                            run_visual_only=run_visual_only) 
        else:
            test_loss, test_acc = 0.0, 0.0

       
        print(
            f"Epoch: {epoch} | LR: {lr:.2e} | "
            f"Train loss: {train_loss:.4f} | "
            f"Test loss: {test_loss:.4f} | "
            f"Test acc: {test_acc * 100:.2f}%"
        )

        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results



def model_evaluation(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader, 
                     y_test, 
                     device=device,
                     run_visual_only=False): 
 
    model.eval()
    y_pred = []
    y_prob = []
    
    y_true_collected = [] 

    with torch.inference_mode():
        for img, input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluating"):
            img = img.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
           
            logits = model.forward_cluster(
                img, 
                input_ids, 
                attention_mask,
                run_visual_only=run_visual_only
            )
            
            preds = logits.argmax(dim=1).cpu().numpy()
            probs = logits.cpu().numpy()
            
            y_pred.extend(preds)
            y_prob.extend(probs)
            y_true_collected.extend(labels.numpy())

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    target_labels = np.array(y_true_collected)

    ind, acc = utils.cluster_acc(target_labels, y_pred)
    
    return y_prob, y_pred, ind, acc