#!/usr/bin/env python3
"""Run board_diag.py on a remote board through SSH from the Windows/Linux host."""
import argparse
import paramiko


def main():
    parser = argparse.ArgumentParser(description="Run /home/cat/board_diag.py over SSH")
    parser.add_argument("--host", default="10.138.103.190")
    parser.add_argument("--user", default="cat")
    parser.add_argument("--password", default="2335")
    args = parser.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=10)
    stdin, stdout, stderr = ssh.exec_command("python3 /home/cat/board_diag.py", timeout=180)
    print(stdout.read().decode("utf-8", errors="replace"))
    err = stderr.read().decode("utf-8", errors="replace")
    if err:
        print("ERR:", err)
    ssh.close()


if __name__ == "__main__":
    main()
