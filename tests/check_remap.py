# testing program for remap.cpp
import sys
def read_input():
	d = {}
	for l in sys.stdin:
		sp = l.split()
		rank = int(sp[0])
		row = int(sp[1])
		col = int(sp[2])
		x = int(sp[3])
		y = int(sp[4])
		z = int(sp[5])
		a = int(sp[6])
		b = int(sp[7])
		c = int(sp[8])
		if (row,col) in d:
			print("error 0", row, col)
		d[(row,col)] = (x,y,z,a,b,c)
	return d

def check2(t, s):
	x = abs(t[0] - s[0])
	y = abs(t[1] - s[1])
	z = abs(t[2] - s[2])
	a = abs(t[3] - s[3])
	b = abs(t[4] - s[4])
	c = abs(t[5] - s[5])
	return x+y+z+a+b+c == 1

rows=528
cols=240

dic = read_input()
for r in range(0, rows):
	for c in range(0, cols):
		if not (r,c) in dic:
			print("errorr 1", r, c)
for k in dic.keys():
	if k[0] == 0:
		lt = rows-1
	else:
		lt = k[0] - 1
	if k[0] == rows-1:
		rt = 0
	else:
		rt = k[0] + 1
	if k[1] == cols-1:
		dn = 0
	else:
		dn = k[1] + 1
	if k[1] == 0:
		up = cols-1 
	else:
		up = k[1] - 1
	center = dic[k]
	north = dic[(k[0], up)]
	south = dic[(k[0], dn)]
	east = dic[(rt, k[1])]
	west = dic[(lt, k[1])]
	if not check2(center, north) or not check2(center, south) or not check2(center, east) or not check2(center, west):
		print("error", center, north, south, east, west)
		sys.exit(1)
print("ok")
