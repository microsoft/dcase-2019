from datetime import datetime
import subprocess

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
NUM_COARSE_LABELS = 8

for i in range(NUM_COARSE_LABELS):
    # using subprocess since pytorch is leaking GPU RAM after each run.
    subprocess.run(['python', 'train_branches.py', str(i), DATE])
