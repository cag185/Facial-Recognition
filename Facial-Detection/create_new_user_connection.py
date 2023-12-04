import subprocess
import sys
import os

folder_name = sys.argv[0]

# Replace '/path/to/venv' with the actual path to your virtual environment
venv_path = '~/Desktop/Facial-Recognition/venv'

# Activate the virtual environment directly without using source
activate_command = os.path.join(venv_path, 'bin', 'activate')
exec(open(activate_command).read(), {'__file__': activate_command})

# Run the Python script
# Replace with the actual path to your Python script
script_to_run = '~/Desktop/Facial-Recognition/Facial-Detection/new_user_folder_init.py'
python_command = f'python {script_to_run} {folder_name}'

# Combine commands and execute
full_command = f'{python_command}'

try:
    subprocess.run(full_command, shell=True, check=True)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    sys.exit(1)
