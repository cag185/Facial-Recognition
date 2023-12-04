import subprocess
import sys
import os

folder_name_to_create = sys.argv[0]

# Replace '/path/to/venv' with the actual path to your virtual environment
venv_path = '~/Desktop/Facial-Recognition/venv/bin/activate'

# Run the Python script
# Replace with the actual path to your Python script
script_to_run = '~/Desktop/Facial-Recognition/Facial-Detection/'

# Activate the virtual environment directly without using source
activate_command = venv_path
exec(open(activate_command).read(), {'__file__': activate_command})

# Run the Python script
python_command = f'python {script_to_run} {folder_name_to_create}'

# Combine commands and execute
full_command = f'{python_command}'

try:
    subprocess.run(full_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    sys.exit(1)
