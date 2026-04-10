import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image

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
            iou_list.append(float('nan')) 
        else:
            iou_list.append(intersection / union)
            
        dice_denominator = pred_inds.sum().item() + target_inds.sum().item()
        if dice_denominator == 0:
            dice_list.append(float('nan'))
        else:
            dice_list.append((2. * intersection) / dice_denominator)
            
    return iou_list, dice_list

class HouseDataset(Dataset):
    def __init__(self, split="train", transform=None):
        print(f"Loading {split} dataset from Hugging Face Parquet...")
        
        # Bypass the broken Hugging Face script and load the auto-converted Parquet files directly
        parquet_url = f"hf://datasets/keremberke/satellite-building-segmentation@refs/convert/parquet/full/{split}/*.parquet"
        
        # We use split="train" here because when loading direct data files, 
        # Hugging Face defaults to putting them in a "train" bucket, regardless of the folder name.
        self.dataset = load_dataset("parquet", data_files=parquet_url, split="train[:50]")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    # Week 7 Mask Generation Function
    def make_mask(self, labelled_bbox, image):
        x_min_ones, y_min_ones, width_ones, height_ones = labelled_bbox
        x_min_ones, y_min_ones, width_ones, height_ones = int(x_min_ones), int(y_min_ones), int(width_ones), int(height_ones)
        
        mask_instance = np.zeros((image.width, image.height))
        last_x = x_min_ones + width_ones
        last_y = y_min_ones + height_ones
        mask_instance[x_min_ones:last_x, y_min_ones:last_y] = np.ones((int(width_ones), int(height_ones)))
        return mask_instance.T

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        
        # Start with a blank background mask (all zeros)
        combined_mask = np.zeros((image.height, image.width))
        
        # Combine all house bounding boxes in this image into the mask
        for bbox in item["objects"]["bbox"]:
            mask_instance = self.make_mask(bbox, image)
            # Use np.maximum to combine overlapping house masks
            combined_mask = np.maximum(combined_mask, mask_instance)
        
        # Convert mask to PIL Image so torchvision transforms can resize it easily
        mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
        
        if self.transform:
            # We must apply the SAME transform (like resizing/cropping) to both the image and the mask
            # so the pixels still line up.
            image = self.transform(image)
            
            # Mask needs specific handling: no normalization, and resized using nearest neighbor 
            # to keep it strictly 0 (background) and 1 (house)
            mask_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])
            mask_img = mask_transform(mask_img)
            
        # Convert mask to a PyTorch tensor of class indices (0 or 1)
        mask_tensor = torch.tensor(np.array(mask_img) > 0, dtype=torch.long)
        
        return image, mask_tensor

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # 1. Initialize your Datasets and DataLoaders
    # The transform matches the Preprocess steps expected by DeepLabV3 in app.py
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Note: To save time testing, you might want to use a subset of the data first
    train_dataset = HouseDataset(split="train", transform=preprocess)
    val_dataset = HouseDataset(split="validation", transform=preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 2. Setup Model 
    # num_classes=2 assuming Background (0) and House (1)
    model = models.segmentation.deeplabv3_mobilenet_v3_large(num_classes=2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 3. Training Loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        running_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Training Loss: {running_loss/len(train_loader):.4f}")
            
        # 4. Evaluation Loop
        model.eval()
        total_iou, total_dice, batches = 0, 0, 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                
                iou, dice = calculate_iou_and_dice(outputs, masks, num_classes=2)
                
                # We only care about class '1' (the house) metrics
                if not torch.isnan(torch.tensor(iou[1])):
                    total_iou += iou[1]
                    total_dice += dice[1]
                    batches += 1
                    
        if batches > 0:
            print(f"Val IoU: {total_iou/batches:.4f} | Val Dice: {total_dice/batches:.4f}")

    # 5. Save the trained weights
    print("Saving model weights to house_model.pth...")
    torch.save(model.state_dict(), 'house_model.pth')
    print("Training complete.")

if __name__ == "__main__":
    main()