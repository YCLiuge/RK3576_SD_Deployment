#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
pip install --user paramiko -q 2>/dev/null

python3 << 'PYEOF'
import paramiko
HOST='10.138.103.190'; USER='cat'; PASS='2335'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=22, username=USER, password=PASS, timeout=10)
sftp = ssh.open_sftp()

LOCAL = '/home/lzy0x91f/lcm_sd'
for f in ['unet/model_256.rknn', 'vae_decoder/model_256.rknn']:
    local = f'{LOCAL}/{f}'
    remote = f'/home/cat/lcm_sd/{f}'
    print(f'Uploading {f}...')
    sftp.put(local, remote)

sftp.close()
ssh.close()
print('Done')
PYEOF
