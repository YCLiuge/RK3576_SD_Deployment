#!/usr/bin/env python3
"""Upload board-side scripts and optional cached embeddings to the RK3576 board."""
import argparse
import os
import posixpath

import paramiko


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOARD_FILES = [
    "sd_lcm.py",
    "sd_prompt.py",
    "sd_inference.py",
    "board_lcm.py",
    "board_lcm_256.py",
    "board_gen_emb.py",
    "board_diag.py",
]


def main():
    parser = argparse.ArgumentParser(description="Deploy board Python scripts to /home/cat")
    parser.add_argument("--host", default="10.138.103.190")
    parser.add_argument("--user", default="cat")
    parser.add_argument("--password", default="2335")
    parser.add_argument("--remote-dir", default="/home/cat")
    parser.add_argument("--with-embeds", action="store_true", help="also upload embeds/pos_emb.npy and neg_emb.npy")
    args = parser.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
    sftp = ssh.open_sftp()
    try:
        for name in BOARD_FILES:
            local = os.path.join(ROOT, "board", name)
            remote = posixpath.join(args.remote_dir, name)
            sftp.put(local, remote)
            sftp.chmod(remote, 0o755)
            print(f"uploaded {remote}")

        if args.with_embeds:
            for name in ["pos_emb.npy", "neg_emb.npy"]:
                local = os.path.join(ROOT, "embeds", name)
                remote = posixpath.join(args.remote_dir, name)
                if os.path.exists(local):
                    sftp.put(local, remote)
                    print(f"uploaded {remote}")
    finally:
        sftp.close()
        ssh.close()


if __name__ == "__main__":
    main()
