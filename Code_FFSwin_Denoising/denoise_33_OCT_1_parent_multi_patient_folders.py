import torch, os, cv2, glob
import numpy as np
from tqdm import tqdm
from model_architecture import OCT3DDenoisingAutoencoder

def main():
    # ------------------------------------------------
    #  Set input and output directories where each folder may contains multiple patient folders 
    # ------------------------------------------------
   
    IN_DIR = r"33_Extracted_OCT_data\WetAMD"
    OUT_DIR = r"33_Extracted_OCT_data_denoised\WetAMD"

    CP_PATH = r"models\denoiser.pth"

    if os.path.exists(CP_PATH):
        print("Model Path found...")
    else:
        print(" Model checkpoint not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference on {device}...")

    # Load Model
    model = OCT3DDenoisingAutoencoder().to(device)

    print(f"Loading weights from {CP_PATH}")
    cp = torch.load(CP_PATH, map_location=device)

    if isinstance(cp, dict) and 'model_state_dict' in cp:
        model.load_state_dict(cp['model_state_dict'])
    else:
        model.load_state_dict(cp)

    model.eval()

    print(f"Processing patient folders from: {IN_DIR}")

    # ------------------------------------------------
    # LOGIC: AUTO-DETECT PATIENT FOLDERS
    # ------------------------------------------------
    patient_folders = [
        f.path for f in os.scandir(IN_DIR)
        if f.is_dir()
    ]

    for p in tqdm(patient_folders, desc="Patients"):
        files = sorted(glob.glob(os.path.join(p, "*.bmp")))
        vol = []

        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                vol.append(cv2.resize(img, (256, 256)))

        if not vol:
            continue

        original_depth = len(vol)

        # Pad depth to even number
        if len(vol) % 2 != 0:
            vol.append(vol[-1])

        t = torch.FloatTensor(np.array(vol)/255.0).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(t)

        out_np = (out.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Remove padded slice
        out_np = out_np[:original_depth]

        # Save location
        rel = os.path.relpath(p, IN_DIR)
        save_path = os.path.join(OUT_DIR, rel)
        os.makedirs(save_path, exist_ok=True)

        # Save output slices
        for i, f in enumerate(files):
            if i < len(out_np):
                cv2.imwrite(os.path.join(save_path, os.path.basename(f)), out_np[i])

    print(f" Inference Complete. Clean data saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
