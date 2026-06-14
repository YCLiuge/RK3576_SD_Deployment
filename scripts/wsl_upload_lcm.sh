#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH

# Install paramiko in WSL if needed
pip install --user paramiko -q 2>/dev/null

python3 << 'PYEOF'
import paramiko, os

HOST='10.138.103.190'; USER='cat'; PASS='2335'
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=22, username=USER, password=PASS, timeout=10)
sftp = ssh.open_sftp()

base = '/home/cat/lcm_sd'
LOCAL = '/home/lzy0x91f/lcm_sd'

# Create dirs
for d in ['', '/text_encoder', '/unet', '/vae_decoder', '/scheduler', '/tokenizer']:
    try: sftp.mkdir(base + d)
    except: pass

# Upload RKNN models
for f in ['text_encoder/model.rknn', 'unet/model.rknn', 'vae_decoder/model.rknn']:
    local = os.path.join(LOCAL, f)
    remote = f'{base}/{f}'
    sz = os.path.getsize(local) / 1024**2
    print(f'Uploading {f} ({sz:.0f} MB)...')
    sftp.put(local, remote)

# Upload scheduler + tokenizer
for f in ['scheduler/scheduler_config.json']:
    sftp.put(os.path.join(LOCAL, f), f'{base}/{f}')
    print(f'Uploaded {f}')

for f in ['merges.txt', 'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json']:
    sftp.put(os.path.join(LOCAL, 'tokenizer', f), f'{base}/tokenizer/{f}')
    print(f'Uploaded tokenizer/{f}')

# Upload runtime script
sftp.put(os.path.join(LOCAL, 'run_rknn-lcm.py'), f'{base}/run_rknn-lcm.py')
print('Uploaded run_rknn-lcm.py')

sftp.close()
ssh.close()
print('Done!')
PYEOF
