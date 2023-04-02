import sys
from math import log
import shutil
from pathlib import Path
from datetime import datetime
from threading import Thread
from subprocess import Popen, PIPE
from concurrent.futures import ThreadPoolExecutor, as_completed

N = 100

scores = [30.0] * N


def read_stream(name, in_file, out_file):
    for line in in_file:
        print(f"[{name}] {line.strip()}", file=out_file)
        try:
            scores[name] = log(int(line.strip().split()[-1]))
        except:
            pass


def run(cmd, name, timeout=None):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    stdout_thread = Thread(target=read_stream, args=(name, proc.stdout, sys.stdout))
    stderr_thread = Thread(target=read_stream, args=(name, proc.stderr, sys.stderr))
    stdout_thread.start()
    stderr_thread.start()
    try:
        proc.wait(timeout=timeout)
    except TimeoutError:
        print(f"[{name}] Timeout!", file=sys.stderr)
        proc.kill()
    return proc


Path("out").mkdir(exist_ok=True)
out_dir = Path("out") / datetime.now().isoformat()
out_dir.mkdir()
with ThreadPoolExecutor(4) as executor:
    futures = []
    for i in range(N):
        in_file = f"./tools/in/{i:04d}.txt"
        out_file = out_dir / f"{i:04d}.txt"
        cmd = f"./a.out < {in_file} > {out_file} && ./tools/target/release/vis {in_file} {out_file}"
        futures.append(executor.submit(run, cmd, i, 10.0))
    as_completed(futures)

mean_score = sum(scores) / len(scores)

print(f"Mean Score = {mean_score}")

shutil.move(out_dir, f"{out_dir}_{mean_score:.2f}")

exit()
