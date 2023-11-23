# scp_connection.py
# this file is designed to create a connection to the Virtual Machine and attempt to transfer images to the VM

# IP address string
import subprocess
destination = "cag185@172.208.16.5:~/Facial-Recognition/Test-Images/testImage.png"
source = "/home/molay/Desktop/Facial-Recognition/Test-Images/testImage.png"
private_key_loc = "/home/molay/private_vm_key/Facial-Detection-VM_key.pem"
# create the scp connection
# run the command


scp_command = ["scp", "-i", private_key_loc, source, destination]

try:
    subprocess.run(scp_command, check=True)
    print("File transfer successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error transferring file(s): {e}")
