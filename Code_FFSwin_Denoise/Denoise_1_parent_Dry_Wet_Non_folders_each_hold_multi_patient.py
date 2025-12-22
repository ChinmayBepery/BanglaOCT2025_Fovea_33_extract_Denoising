import torch, os, cv2, glob
import numpy as np
from tqdm import tqdm
from model_architecture import OCT3DDenoisingAutoencoder

def main():
    IN_DIR = r"33_Extracted_OCT_data"
    OUT_DIR = r"33_Extracted_OCT_data_denoised"
    
    # Try loading final model first, then checkpoint
    CP_PATH = "models\denoiser.pth"
    if not os.path.exists(CP_PATH):
        CP_PATH = "models\denoiser.pth"
    else: 
        print(f"Model Path found...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on {device}...")
    
    # Initialize Model
    model = OCT3DDenoisingAutoencoder().to(device)
    
    if os.path.exists(CP_PATH):
        print(f"Loading weights from {CP_PATH}")
        cp = torch.load(CP_PATH, map_location=device, weights_only=True)
        # Handle cases where checkpoint is a dict vs just weights
        if isinstance(cp, dict) and 'model_state_dict' in cp:
            model.load_state_dict(cp['model_state_dict'])
        else:
            model.load_state_dict(cp)
    else:
        print(" Model not found! Please train the denoiser first.")
        return

    model.eval()
    
    print(f"Processing data from {IN_DIR}...")
    
    for cat in ['DryAMD', 'WetAMD', 'NonAMD']:
        path = os.path.join(IN_DIR, cat)
        if not os.path.exists(path): continue
        
        patients = [f.path for f in os.scandir(path) if f.is_dir()]
        
        # Loop through patients
        for p in tqdm(patients, desc=cat):
            files = sorted(glob.glob(os.path.join(p, "*.bmp")))
            vol = []
            
            # Load images
            for f in files:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None: 
                    # Resize is crucial if your model expects 256x256
                    vol.append(cv2.resize(img, (256, 256)))
            
            if not vol: continue
            
            # --- FIX: PAD DEPTH TO BE EVEN ---
            # Swin Transformer fails if depth is odd (e.g. 33). We pad to 34.
            original_depth = len(vol)
            if len(vol) % 2 != 0:
                vol.append(vol[-1]) # Duplicate last slice
            
            # Convert to Tensor (Batch, Channel, Depth, Height, Width)
            t = torch.FloatTensor(np.array(vol)/255.0).unsqueeze(0).unsqueeze(0).to(device)
            
            # Denoise
            with torch.no_grad():
                out = model(t)
            
            # Convert back to Numpy
            out_np = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)
            
            # --- FIX: CROP BACK TO ORIGINAL DEPTH ---
            # Remove the fake padding slice so output matches input exactly
            out_np = out_np[:original_depth]
            
            # Setup Save Path
            rel = os.path.relpath(p, IN_DIR)
            save_path = os.path.join(OUT_DIR, rel)
            os.makedirs(save_path, exist_ok=True)
            
            # Save slices using original filenames
            for i, f in enumerate(files):
                if i < len(out_np):
                    cv2.imwrite(os.path.join(save_path, os.path.basename(f)), out_np[i])

    print(f" Inference Complete. Clean data saved to {OUT_DIR}")

if __name__ == "__main__": main()
