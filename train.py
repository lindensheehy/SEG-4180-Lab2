import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def calculate_iou_and_dice(pred, target, num_classes):
    """Calculates Intersection over Union (IoU) and Dice Score."""
    pred = torch.argmax(pred, dim=1)
    iou_list, dice_list = [], []
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        
        if union == 0:
            iou_list.append(float('nan'))  # Ignore if class not in ground truth
        else:
            iou_list.append(intersection / union)
            
        dice_denominator = pred_inds.sum().item() + target_inds.sum().item()
        if dice_denominator == 0:
            dice_list.append(float('nan'))
        else:
            dice_list.append((2. * intersection) / dice_denominator)
            
    return iou_list, dice_list

def main():
    # 1. TODO: Insert Week 7 Pixel Mask Generation and Dataloader here
    # train_loader = ...
    # val_loader = ...
    
    # 2. Setup Model (e.g., DeepLabV3 or UNet)
    model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        # for images, masks in train_loader:
            # images, masks = images.to(device), masks.to(device)
            # optimizer.zero_grad()
            # outputs = model(images)['out']
            # loss = criterion(outputs, masks)
            # loss.backward()
            # optimizer.step()
            
        # 4. Evaluation Loop (Calculate IoU and Dice on Val/Test set)
        model.eval()
        total_iou = 0
        total_dice = 0
        batches = 0
        with torch.no_grad():
            # for images, masks in val_loader:
                # outputs = model(images.to(device))['out']
                # iou, dice = calculate_iou_and_dice(outputs, masks.to(device), num_classes=2)
                # total_iou += iou[1] # Assuming class '1' is the house
                # total_dice += dice[1]
                # batches += 1
            pass
            
        # print(f"Val IoU: {total_iou/batches:.4f} | Val Dice: {total_dice/batches:.4f}")

if __name__ == "__main__":
    main()