import torch, argparse, os
import torch.nn.functional as F
from torch.utils.data import DataLoader
# FIX: Use new AMP import
from torch.amp import GradScaler, autocast 
from model_architecture import OCT3DDenoisingAutoencoder
from dataset import BalancedOCTVolumeDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r"D:\OCTData\BanglaOCT2025_Dataset\data")
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Denoiser on {device}...")
    
    # 1. Load Dataset
    ds = BalancedOCTVolumeDataset(args.data_dir)
    # 2. DataLoader
    # Note: If you get "Shared Memory" errors, set num_workers=0
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    
    # 3. Model
    model = OCT3DDenoisingAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # FIX: Scaler for CUDA
    scaler = GradScaler('cuda')
    
    model.train()
    print("Starting Training Loop...")
    
    for ep in range(args.epochs):
        loss_avg = 0
        for i, clean in enumerate(dl):
            clean = clean.to(device)
            
            # Generate Unsupervised Noise
            noisy = torch.clamp(clean + torch.randn_like(clean)*0.15, 0, 1) 
            
            opt.zero_grad()
            
            # FIX: Autocast with device_type='cuda'
            with autocast('cuda'):
                out = model(noisy)
                # Loss: MSE + L1 for sharpness
                loss = F.mse_loss(out, clean) + 0.1 * torch.mean(torch.abs(out - clean)) 
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            loss_avg += loss.item()
            
        print(f"Epoch {ep+1}/{args.epochs}, Loss: {loss_avg/len(dl):.5f}")
        
    torch.save(model.state_dict(), "models\\denoiser.pth")
    print("Denoiser Saved Successfully.")

if __name__ == "__main__": main()
