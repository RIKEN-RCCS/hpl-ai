#! /usr/bin/python
import subprocess

print("matrix-size block-size num-procs num-procs-row error-gemv error-trf")
for bs in range(50, 100, 23):
	for nb in range(5, 21, 1):
		n = bs * nb 
		for p in range(2, 21, 1):
			for nrow in range(1, 21, 1):
				if p%nrow != 0: continue
				for t in ["-t", "-not"]:
					for d in ["-d", "-nod"]:
						cmd = "mpirun --oversubscribe -n {0} ./a.out {1} {2} {3} {4} {5}".format(p, n, bs, nrow, t, d)
						subprocess.check_call(cmd, shell=True)
