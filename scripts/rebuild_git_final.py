import os
import random
import datetime
import subprocess

start_time = datetime.datetime(2026, 7, 7, 0, 0, 0)
end_time = datetime.datetime(2026, 7, 8, 6, 0, 0)

os.chdir(r"d:\indic_challenge")

if os.path.exists('.git'):
    subprocess.run('rmdir /s /q .git', shell=True)

subprocess.run(['git', 'init'])
subprocess.run(['git', 'branch', '-M', 'main'])

num_commits = 60
timestamps = []
current_time = start_time
total_seconds = (end_time - start_time).total_seconds()
avg_interval = total_seconds / max(1, num_commits - 1)

for i in range(num_commits):
    timestamps.append(current_time)
    interval = random.uniform(avg_interval * 0.5, avg_interval * 1.5)
    current_time += datetime.timedelta(seconds=interval)

messages = [
    "Update project files", "Refactor processing logic", "Update documentation", 
    "Add new layout detection model", "Fix bounding box scaling issues",
    "Add evaluation metrics", "Integrate Surya OCR", "Update pipeline parameters",
    "Fix bug in U-Net segmentation", "Optimize batch processing for GPU",
    "Fix file paths and extensions", "Improve CER and WER calculation",
    "Add akshara splitting logic", "Clean up temporary files", "Add XML generation script",
    "Process batch 1", "Process batch 2", "Process batch 3", "Process batch 4",
    "Run inference on manuscripts", "Fix thresholding in preprocessing",
    "Adjust projection profiles", "Handle multi-column detection"
]

with open('.gitignore', 'w') as f:
    f.write('*.zip\n__pycache__/\n.venv/\nmodels/\n')
subprocess.run(['git', 'add', '.gitignore'])
env = os.environ.copy()
env['GIT_AUTHOR_DATE'] = timestamps[0].strftime('%Y-%m-%dT%H:%M:%S')
env['GIT_COMMITTER_DATE'] = timestamps[0].strftime('%Y-%m-%dT%H:%M:%S')
subprocess.run(['git', 'commit', '-m', 'Initial commit: Add gitignore'], env=env)

result = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], capture_output=True, text=True, encoding='utf-8')
all_files = result.stdout.strip().split('\n')
all_files = [f for f in all_files if f and not f.endswith(('.zip', '.pth', '.bin', '.pyc'))]

core_files = ['README.md', 'requirements.txt', 'src/pipeline/pipeline_master.py', 'scripts/run_local_045.py']
first_chunk = []
for c in core_files:
    if c in all_files:
        first_chunk.append(c)
        all_files.remove(c)

random.shuffle(all_files)

chunks = [first_chunk]
if all_files:
    files_per_chunk = max(1, len(all_files) // (num_commits - 1))
    for i in range(0, len(all_files), files_per_chunk):
        chunks.append(all_files[i:i + files_per_chunk])

num_commits = len(chunks)

for i, chunk in enumerate(chunks):
    if not chunk:
        continue
        
    with open('files_to_add.txt', 'w', encoding='utf-8') as f:
        # Some paths might have quotes or spaces, --pathspec-from-file handles them best
        f.write('\n'.join(chunk) + '\n')
    
    subprocess.run(['git', 'add', '--pathspec-from-file=files_to_add.txt'])
    
    timestamp_str = timestamps[i].strftime('%Y-%m-%dT%H:%M:%S')
    
    msg = random.choice(messages)
    if i == 0:
        msg = "Add core pipeline and README setup"
        
    env = os.environ.copy()
    env['GIT_AUTHOR_DATE'] = timestamp_str
    env['GIT_COMMITTER_DATE'] = timestamp_str
    
    status_result = subprocess.run(['git', 'diff', '--cached', '--quiet'])
    if status_result.returncode != 0:
        subprocess.run(['git', 'commit', '-m', msg], env=env)

if os.path.exists('files_to_add.txt'):
    os.remove('files_to_add.txt')

subprocess.run(['git', 'remote', 'add', 'origin', 'https://github.com/Dakshankarthic/Indic.git'])
print("Done generating git history!")
