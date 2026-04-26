import schedule
import time
import subprocess
import sys
import os
import logging
import psutil

# Setup logging
logging.basicConfig(
    filename='scheduler.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

LOCK_FILE = "scheduler.lock"

def is_stale_lock(lock_path):
    try:
        pid = int(open(lock_path).read().strip())
        return not psutil.pid_exists(pid)
    except Exception:
        return True

def job():
    logging.info("Triggered automated MLOps pipeline job.")
    
    if os.path.exists(LOCK_FILE):
        if is_stale_lock(LOCK_FILE):
            logging.warning("Found a stale lock file from a previous crash. Removing it.")
            os.remove(LOCK_FILE)
        else:
            logging.warning("Lock file exists and process is running! Another pipeline run is executing. Skipping.")
            return
        
    try:
        # Create lock with current process ID
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
            
        logging.info("Starting run_pipeline.py...")
        result = subprocess.run([sys.executable, "run_pipeline.py"], capture_output=True, text=True)
        
        if result.returncode != 0:
            logging.error(f"Pipeline failed with exit code {result.returncode}.")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            # In a real system, you would trigger an email or Slack webhook here
        else:
            logging.info("Pipeline completed successfully.")
            
    except Exception as e:
        logging.error(f"Exception during pipeline execution: {e}", exc_info=True)
    finally:
        # Remove lock
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

# Schedule the pipeline to run every day at 2:00 AM (simulating cron)
schedule.every().day.at("02:00").do(job)
schedule.every(12).hours.do(job)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("🤖 Enterprise MLOps Scheduler started.")
    print("Logs are written to scheduler.log")
    logging.info("Scheduler started.")
    
    if os.path.exists(LOCK_FILE) and is_stale_lock(LOCK_FILE):
        os.remove(LOCK_FILE)
        logging.info("Removed stale lock file on startup.")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nScheduler stopped.")
        logging.info("Scheduler stopped by user.")
