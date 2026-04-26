import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n=============================================")
    print(f"🚀 Running: {script_name}")
    print(f"=============================================")
    
    # Run the script using the current python executable
    result = subprocess.run([sys.executable, script_name])
    
    if result.returncode != 0:
        print(f"\n❌ Error: {script_name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    
    print(f"✅ Finished: {script_name} successfully.\n")

if __name__ == "__main__":
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    pipeline_scripts = [
        "src/preprocess.py",
        "src/train.py",
        "src/evaluate.py",
        "src/monitor.py"
    ]
    
    print("Starting the Enterprise Traffic Volume Prediction MLOps Pipeline...")
    
    for script in pipeline_scripts:
        run_script(script)
        
    print("🎉 All MLOps pipeline stages completed successfully!")
