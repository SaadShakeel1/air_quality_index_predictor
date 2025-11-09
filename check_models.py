"""
Helper script to check if model files exist
Run this before starting the Streamlit app
"""
from pathlib import Path

MODEL_DIR = Path("notebook/models")
REQUIRED_FILES = [
    "final_classifier.pkl",
    "final_regressor.pkl",
    "scaler.pkl",
    "reg_scaler.pkl"
]

def check_models():
    """Check if all required model files exist"""
    print("=" * 50)
    print("Model Files Check")
    print("=" * 50)
    
    all_exist = True
    missing_files = []
    
    for file in REQUIRED_FILES:
        file_path = MODEL_DIR / file
        exists = file_path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"{status} {file}")
        
        if not exists:
            all_exist = False
            missing_files.append(file)
    
    print("=" * 50)
    
    if all_exist:
        print("[SUCCESS] All model files found! You can run the Streamlit app.")
        print("\nTo start the app, run:")
        print("  streamlit run app.py")
    else:
        print("[ERROR] Missing model files! Please train the models first.")
        print("\nTo train models:")
        print("  1. Open notebook/Model_Training.ipynb in Jupyter")
        print("  2. Run all cells")
        print("  3. Models will be saved to notebook/models/")
        print(f"\nMissing files: {', '.join(missing_files)}")
    
    return all_exist

if __name__ == "__main__":
    check_models()

