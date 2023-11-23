# the purpose of this file is to attempt to create an ssh connection
# from the raspberrypi to the azure VM and copy files from pi -> vm

from paramiko import SSHClient
from scp import SCPClient

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect("172.208.16.5")

# SCPCLient takes a paramiko transport as an argument
scp = SCPClient(ssh.get_transport())

scp.put('test.txt', 'test2.txt')
scp.get('test2.txt')

# Uploading the 'test' directory with its content in the
# '/home/user/dump' remote directory
scp.put('test', recursive=True, remote_path='~/')

scp.close()
