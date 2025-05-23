import shutil
import datetime
import os

def backup_training_data():
    """Backup all training data and configs (Placeholder - ensure directories exist or handle errors)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("backups", f"training_backup_{timestamp}")
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Created backup directory: {backup_dir}")

        data_src = "data"
        data_dest = os.path.join(backup_dir, "data")
        if os.path.exists(data_src) and os.path.isdir(data_src):
            shutil.copytree(data_src, data_dest)
            print(f"Copied '{data_src}' to '{data_dest}'.")
        else:
            print(f"Warning: '{data_src}' directory not found, skipping backup.")

        src_src = "src"
        src_dest = os.path.join(backup_dir, "src")
        if os.path.exists(src_src) and os.path.isdir(src_src):
            shutil.copytree(src_src, src_dest)
            print(f"Copied '{src_src}' to '{src_dest}'.")
        else:
            print(f"Warning: '{src_src}' directory not found, skipping backup.")
        
        env_file = ".env"
        env_dest = os.path.join(backup_dir, ".env")
        env_example_file = ".env.example"
        env_example_dest = os.path.join(backup_dir, ".env.example")

        if os.path.exists(env_file):
            shutil.copy(env_file, env_dest)
            print(f"Copied '{env_file}' to '{env_dest}'.")
        elif os.path.exists(env_example_file):
            shutil.copy(env_example_file, env_example_dest)
            print(f"Copied '{env_example_file}' to '{env_example_dest}' as '{env_file}' was not found.")
        else:
            print(f"Warning: Neither '{env_file}' nor '{env_example_file}' found, skipping backup.")
        
        print(f"✅ Backup simulation complete: {backup_dir}")
    except Exception as e:
        print(f"Error during backup simulation: {e}")


if __name__ == "__main__":
    pass
