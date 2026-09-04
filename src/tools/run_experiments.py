import os
import random
import subprocess
import csv
from datetime import datetime
from itertools import product

def main():
    # Configuration
    CHARACTERS_DIR = "data/characters"
    situation_path_list = [
        ("reference/situations/angry.yaml", "prompts/agressive_3.txt"), 
        ("reference/situations/anxious.yaml", "prompts/anxious.txt")
        ]
    TURNS = 15
    
    # Setup output directory
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = f"results_{date_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all character files
    character_files = [f for f in os.listdir(CHARACTERS_DIR) if f.endswith(".txt")]
    character_files = random.sample(character_files, 3)
    reasoners = ["random", "dummy"]
    models = ["gpt-oss-20b", "gpt-oss-120b"]
    metadata = []
    
    experiments = list(product(models, character_files, reasoners, situation_path_list))
    
    random.shuffle(experiments)
    
    for i, (model, char_file, reasoner, (situation_path, prompt_path)) in enumerate(experiments):
        char_path = os.path.join(CHARACTERS_DIR, char_file)
        char_name = os.path.splitext(char_file)[0]
        
        filename = f"dialogue_{i}.txt"
        output_file = os.path.join(output_dir, filename)
        
        # Build command
        cmd = [
            "python3", "simulate_session.py",
            "--reasoner", reasoner,
            "--turns", str(TURNS),
            "--situations_path", situation_path,
            "--client_prompt_path", prompt_path,
            "--character_path", char_path,
            "--output_file", output_file,
            "--client_base_model", model
        ]
        
        print(f"  Run experiment {i} out of {len(experiments)} for {char_name} (reasoner={reasoner}, situation_path={situation_path}), client_base_model={model}...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"  Error running simulation")
            continue
        
        # Store metadata
        metadata.append({
            "character": char_name,
            "client_base_model": model,
            "reasoner": reasoner,
            "turns": TURNS,
            "situations_path": situation_path,
            "prompt_path": prompt_path,
            "filename": filename
        })

    # Write metadata to CSV
    csv_file = os.path.join(output_dir, "metadata.csv")
    if metadata:
        keys = metadata[0].keys()
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata)
            
    print(f"\nAll runs completed. Results and metadata saved in {output_dir}")

if __name__ == "__main__":
    main()
