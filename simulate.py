import simpy
import math
import sys
import time
import array
size_small = 1024
lat_per_hop = 200
comm_throughput = 3e9
mem_throughput = 1024 * 1e9
comp_throughput_single = 48 * 64 * 2e9
comp_throughput_half = 48 * 128 * 2e9
chunksize = 256*1024
offloading_latency = 1e5

gemm_ratio = 0.956
trf_ratio = 0.016
trsm_l_ratio = 0.15
trsm_r_ratio = 0.2
#trf_ratio = gemm_ratio
#trsm_l_ratio = gemm_ratio
#trsm_r_ratio = gemm_ratio
li_ratio = 0.4
conv_l_ratio = 0.4
conv_r_ratio = 0.35
class ProcComm:
	def __init__(self, env, id, stores):
		self.env = env
		self.id = id
		self.np = len(stores)
		self.stores = stores
	def latency(self, other):
		left = min(self.id, other)
		right = max(self.id, other)
		nhop = min(right - left, left+len(self.stores)-right)
		return nhop * lat_per_hop
	def recv_impl(self, size, id_from):
		if size < size_small:
			_ = yield self.stores[self.id].get()
		else:
			yield self.env.timeout(self.latency(id_from))
			yield self.stores[id_from].put((size, self.id))
			_ = yield self.stores[self.id].get()
	def send_impl(self, size, id_to):
		if size < size_small:
			yield env.timeout(self.latency(id_to)+size/comm_throughput*1e9)
			yield self.stores[id_to].put((size, self.id))
		else:
			_ = yield self.stores[self.id].get()
			yield env.timeout(self.latency(id_to)+size/comm_throughput*1e9)
			yield self.stores[id_to].put((size, self.id))
	def bcast(self, size, id_root):
		if self.np == 0:
			yield env.timeout(0)
		prev = self.id-1 if self.id > 0 else self.np-1
		next = self.id+1 if self.id+1 < self.np else 0
		if size < size_small or size < chunksize:
			if self.id != id_root:
				yield from self.recv_impl(size, prev)
			if next != id_root:
				yield from self.send_impl(size, next)
		else:
			if self.id != id_root:
				yield env.timeout(self.latency(prev))
				_ = yield self.stores[self.id].get()
				_ = yield self.stores[self.id].get()
				_ = yield self.stores[self.id].get()
			if next != id_root:
				yield self.stores[next].put(0)
				yield self.stores[next].put(0)
				yield env.timeout(self.latency(next)+chunksize/comm_throughput*1e9)
				yield self.stores[next].put(self.id)
			yield env.timeout((size-chunksize)/comm_throughput*1e9)
class ProcComp:
	def __init__(self, env, row, col, nrow, ncol,
		vstores, hstores, nb, nprow, npcol, b, non_blocking=False):
		self.env = env
		self.row = row
		self.col = col
		self.nrow = nrow
		self.ncol = ncol
		self.id = row+col*nrow
		self.vcomm = ProcComm(env, self.row, vstores)
		self.hcomm = ProcComm(env, self.col, hstores)
		self.nb = nb
		self.nprow = nprow
		self.npcol = npcol
		self.b = b
		self.data = {}
		
		if non_blocking:
			self.mpilatency = offloading_latency
			self.active = env.process(self.run_nonblocking())
		else:
			self.mpilatency = 200
			self.active = env.process(self.run())

	def log(self, tag, beg, end, acc):
		if not tag in self.data:
			self.data[tag] = (array.array('d'), array.array('d'), array.array('Q'))
		self.data[tag][0].append(beg)
		self.data[tag][1].append(end)
		self.data[tag][2].append(acc)

	def print_log(self):
		# stats
		print("### simulation statics proc {:5d}".format(self.id))
		totaltime = self.data['TOTAL'][1][0]
		for tag, v in sorted(self.data.items()):
			sumtime = 0.
			sumacc = 0
			for i, beg in enumerate(v[0]):
				sumtime += v[1][i] - beg
				sumacc += v[2][i]
			if sumtime == 0.: sumtime = 1.
			print("{:12s} : {:e} sec. : {:6.2f} % : {:20d} : {:e} Gop/s".format(
				tag, sumtime, 100.*sumtime/totaltime,
				sumacc, 1e-9*sumacc/sumtime))
		with open('Timerdump.{:0>5d}'.format(self.id), 'wb') as f:
			for tag, v in self.data.items():
				size = len(v[0])
				f.write('bio, {}, {}{}\n'.format(size, 'BEG_', tag).encode('utf-8'))
				v[0].tofile(f)
				f.write('bio, {}, {}{}\n'.format(size, 'END_', tag).encode('utf-8'))
				v[1].tofile(f)
				f.write('bio, {}, {}{}\n'.format(size, 'ACC_', tag).encode('utf-8'))
				v[2].tofile(f)

	def lazy_init(self, nrow, ncol):
		nbytes = 6 * self.b*self.b * nrow * ncol
		beg = self.env.now*1e-9
		yield env.timeout(1./li_ratio * nbytes / mem_throughput * 1e9)
		end = self.env.now*1e-9
		self.log("LAZY_INIT",beg,end,int(nbytes/6))
	def diag_trf(self):
		nflops = self.b*(self.b*(4*self.b-3)+5)/6
		beg = self.env.now*1e-9
		yield env.timeout(1./trf_ratio * nflops / comp_throughput_single * 1e9)
		end = self.env.now*1e-9
		self.log("DIAG_LU",beg,end,int(nflops))
	def trsm_l(self, nrow):
		nflops = nrow*self.b*self.b*self.b
		beg = self.env.now*1e-9
		yield env.timeout(1./trsm_l_ratio * nflops / comp_throughput_single * 1e9)
		end = self.env.now*1e-9
		self.log("TRSM_L",beg,end,int(nflops))
	def trsm_r(self, ncol):
		nflops = ncol*self.b*self.b*self.b
		beg = self.env.now*1e-9
		yield env.timeout(1./trsm_r_ratio * nflops / comp_throughput_single * 1e9)
		end = self.env.now*1e-9
		self.log("TRSM_R",beg,end,int(nflops))
	def convert_l(self, nrow):
		nbytes = 6 * self.b * self.b * nrow
		beg = self.env.now*1e-9
		yield env.timeout(1./conv_l_ratio * nbytes / mem_throughput * 1e9)
		end = self.env.now*1e-9
		self.log("CONV_L",beg,end,int(nbytes/6))
	def convert_r(self, ncol):
		nbytes = 6 * self.b * self.b * ncol
		beg = self.env.now*1e-9
		yield env.timeout(1./conv_r_ratio * nbytes / mem_throughput * 1e9)
		end = self.env.now*1e-9
		self.log("CONV_R",beg,end,int(nbytes/6))
	def gemm(self, nrow, ncol):
		if nrow < 1 or ncol < 1:
			return
		nflops = nrow*self.b*ncol*self.b*self.b*2
		beg = self.env.now*1e-9
		yield self.env.timeout(1./gemm_ratio * nflops / comp_throughput_half * 1e9)
		end = self.env.now*1e-9
		self.log("GEMM_UPDATE",beg,end,int(nflops))
	def bcast_l_impl(self, row):
		beg = self.env.now*1e-9 + 1e-6
		yield from self.vcomm.bcast(4*self.b*self.b, row)
		end = self.env.now*1e-9
		self.log("DIAG_BCAST",beg,end,0)
		self.bcast_done = True
	def bcast_r_impl(self, col):
		beg = self.env.now*1e-9 + 1e-6
		yield from self.hcomm.bcast(4*self.b*self.b, col)
		end = self.env.now*1e-9
		self.log("DIAG_BCAST",beg,end,0)
		self.bcast_done = True
	def bcast_diag(self, row, col, reqs):
		reqs.append(self.env.process(self.bcast_l_impl(row)))
		reqs.append(self.env.process(self.bcast_r_impl(col)))
	def bcast_l(self, row):
		self.bcast_done = False
		return self.env.process(self.bcast_l_impl(row))
	def bcast_r(self, col):
		self.bcast_done = False
		return self.env.process(self.bcast_r_impl(col))

	def bcast_lcol_impl(self, row, root):
		beg = self.env.now * 1e-9 + 1e-6
		yield from self.hcomm.bcast(4*self.b*self.b*(self.nprow-row), root)
		end = self.env.now * 1e-9
		self.log('LCOL_BCAST', beg, end, self.b*self.b*(self.nprow-row))
	def bcast_rrow_impl(self, col, root):
		beg = self.env.now * 1e-9 + 1e-6
		yield from self.vcomm.bcast(4*self.b*self.b*(self.npcol-col), root)
		end = self.env.now * 1e-9
		self.log('RROW_BCAST', beg, end, self.b*self.b*(self.npcol-col))
	def bcast_lcol(self, row, root, reqs):
		if row < self.nprow:
			reqs.append(self.env.process(self.bcast_lcol_impl(row, root)))
	def bcast_rrow(self, col, root, reqs):
		if col < self.npcol:
			reqs.append(self.env.process(self.bcast_rrow_impl(col, root)))
	def wait_all(self, tag, reqs):
		beg = self.env.now*1e-9
		yield simpy.events.AllOf(self.env, reqs)
		end = self.env.now*1e-9
		self.log(tag, beg, end, 0)
		reqs.clear()

	def run(self):
		start_time = time.time()
		yield self.env.timeout(1e-9)
		for k in range(self.nb):
			if k%100 == 0 and self.id == 0:
				print("id={} step={}/{} time={} elappsed={}".format(
					self.id, k, self.nb, self.env.now, time.time()-start_time), file=sys.stderr)
			root_row = k % self.nrow
			root_col = k % self.ncol
			i = k // self.nrow + (1 if root_row > self.row else 0)
			j = k // self.ncol + (1 if root_col > self.col else 0)
			reqs = []
			if root_row == self.row and root_col == self.col:
				yield from self.lazy_init(1, 1)
				yield from self.diag_trf()
				if k == self.nb-1:
					break
				self.bcast_diag(root_row, root_col, reqs)

				if i+1 < self.nprow:
					yield from self.lazy_init(self.nprow-i-1, 1)
					yield from self.trsm_l(self.nprow-i-1)
					yield from self.convert_l(self.nprow-i-1)
					self.bcast_lcol(i+1, root_col, reqs)

				if j+1 < self.npcol:
					yield from self.lazy_init(1, npcol-j-1)
					yield from self.trsm_r(self.npcol-j-1)
					yield from self.convert_r(self.npcol-j-1)
					self.bcast_rrow(j+1, root_row, reqs)
				i += 1
				j += 1
			elif root_row == self.row:
				if k == self.nb - 1: break
				yield from self.lazy_init(1, self.npcol - j)
				p = self.bcast_r(root_col)
				self.wait_all('DIAG_BCAST', [p])

				if j < self.npcol:
					yield from self.lazy_init(1, self.npcol-j)
					yield from self.trsm_r(self.npcol-j)
					yield from self.convert_r(self.npcol-j)
					self.bcast_rrow(j, root_row, reqs)
				i += 1
				if i < self.nprow:
					self.bcast_lcol(i, root_col, reqs)
			elif root_col == self.col:
				if k == self.nb - 1: break
				yield from self.lazy_init(self.nprow-i, 1)
				p = self.bcast_l(root_row)
				self.wait_all('DIAG_BCAST', [p])

				if i < self.nprow:
					yield from self.lazy_init(self.nprow-i, 1)
					yield from self.trsm_l(self.nprow-i)
					yield from self.convert_l(self.nprow-i)
					self.bcast_lcol(i, root_col, reqs)
				j += 1
				if j < self.npcol:
					self.bcast_rrow(j, root_row, reqs)
			else:
				if k == self.nb - 1: break
				if i < self.nprow:
					self.bcast_lcol(i, root_col, reqs)
				if j < self.npcol:
					self.bcast_rrow(j, root_row, reqs)
			yield from self.wait_all("WAIT", reqs)
			yield from self.gemm(self.nprow-i, self.npcol-j)
		self.log("TOTAL",0,self.env.now*1e-9,0)

	def run_nonblocking(self):
		last_time = start_time = time.time()
		diag_precomputed = False
		do_diag_precompute = False
		do_lookahead_bcast = False
		pivreq = []
		lrreq = []
		shrow = 0
		shcol = 0
		yield self.env.timeout(1e-9)
		if 0 == self.row and 0 == self.col:
			yield from self.lazy_init(1, 1)
			yield from self.diag_trf()
			self.bcast_diag(0, 0, pivreq)

			yield from self.lazy_init(self.nprow-1, 1)
			yield from self.trsm_l(self.nprow-1)
			yield from self.convert_l(self.nprow-1)
			self.bcast_lcol(1, 0, lrreq)

			yield from self.lazy_init(1, npcol-1)
			yield from self.trsm_r(self.npcol-1)
			yield from self.convert_r(self.npcol-1)
			self.bcast_rrow(1, 0, lrreq)
			shrow = 1
			shcol = 1
		elif 0 == self.row:
			yield from self.lazy_init(1, self.npcol)
			p = self.bcast_r(0)
			yield from self.wait_all("DIAG_BCAST", [p])

			yield from self.lazy_init(1, self.npcol)
			yield from self.trsm_r(self.npcol)
			yield from self.convert_r(self.npcol)
			self.bcast_rrow(0, 0, lrreq)
			shrow = 1
			self.bcast_lcol(1, 0, lrreq)
		elif 0 == self.col:
			yield from self.lazy_init(self.nprow, 1)
			p = self.bcast_l(0)
			yield from self.wait_all("DIAG_BCAST", [p])

			yield from self.lazy_init(self.nprow, 1)
			yield from self.trsm_l(self.nprow)
			yield from self.convert_l(self.nprow)
			self.bcast_lcol(0, 0, lrreq)
			shcol = 1
			self.bcast_rrow(1, 0, lrreq)
		else:
			self.bcast_lcol(0, 0, lrreq)
			self.bcast_rrow(0, 0, lrreq)
		next_root_row = (1) % self.nrow
		next_root_col = (1) % self.ncol
		if do_lookahead_bcast and self.nb>2:
			if next_root_row == self.row and next_root_col == self.col:
				pass
			elif next_root_row == self.row:
				p = self.bcast_r(next_root_col)
			elif next_root_col  == self.col:
				p = self.bcast_l(next_root_row)

		if pivreq: yield from self.wait_all("DIAG_BCAST", pivreq)
		if lrreq: yield from self.wait_all("WAIT", lrreq)
		

		for k in range(1, self.nb):
			if self.id == 0:
				cur_time = time.time()
				if cur_time - last_time > 2:
					duration = time.time() - start_time
					print("id={} step={}/{} time={:.3f} elappsed={:.3f} {:.5f}".format(
						self.id, k, self.nb, self.env.now*1e-9, duration, self.env.now*1e-9/(duration)), file=sys.stderr)
					last_time = cur_time
			root_row = k % self.nrow
			root_col = k % self.ncol
			next_root_row = (k+1) % self.nrow
			next_root_col = (k+1) % self.ncol
			i = k // self.nrow + (1 if root_row > self.row else 0)
			j = k // self.ncol + (1 if root_col > self.col else 0)
			if root_row == self.row and root_col == self.col:
				if not diag_precomputed:
					yield from self.gemm(1, 1)
					yield from self.lazy_init(1, 1)
					yield from self.diag_trf()
					if k != self.nb-1:
						self.bcast_diag(root_row, root_col, pivreq)

				if i+1 < self.nprow:
					yield from self.lazy_init(self.nprow-i-1, 1)
					yield from self.trsm_l(self.nprow-i-1)
					yield from self.convert_l(self.nprow-i-1)
					self.bcast_lcol(i+1, root_col, lrreq)

				if j+1 < self.npcol:
					yield from self.lazy_init(1, npcol-j-1)
					yield from self.trsm_r(self.npcol-j-1)
					yield from self.convert_r(self.npcol-j-1)
					self.bcast_rrow(j+1, root_row, lrreq)
				shrow += 1
				shcol += 1
				i += 1
				j += 1
			elif root_row == self.row:
				if k == self.nb-1: break
				if not do_lookahead_bcast:
					p = self.bcast_r(root_col)
				yield from self.gemm(1, self.npcol-j)
				yield from self.lazy_init(1, self.npcol-j)
				if True:
					while not p.processed:
						#print(self.bcast_done, p.processed, shrow, self.nprow, self.env.now)
						if shrow == self.nprow:
							yield from self.wait_all('WAIT', [p])
							break
						yield from self.gemm(1, self.npcol-j)
						shrow += 1
				else:
					yield from self.wait_all('WAIT', [p])

				self.bcast_lcol(i+1, root_col, lrreq)
				if j < self.npcol:
					yield from self.trsm_r(self.npcol-j)
					yield from self.convert_r(self.npcol-j)
					self.bcast_rrow(j, root_row, lrreq)
				i += 1
			elif root_col == self.col:
				if k == self.nb-1: break
				if not do_lookahead_bcast:
					p = self.bcast_l(root_row)
				yield from self.gemm(self.nprow-i, 1)
				shcol += 1
				yield from self.lazy_init(self.nprow-i, 1)
				if True:
					while not p.processed:
						if shrow == self.nprow:
							yield from self.wait_all('WAIT', [p])
							break
						yield from self.gemm(1, self.npcol-j)
						shrow += 1
				else:
					yield from self.wait_all('WAIT', [p])

				self.bcast_rrow(j+1, root_row, lrreq)
				if i < self.nprow:
					yield from self.trsm_l(self.nprow-i)
					yield from self.convert_l(self.nprow-i)
					self.bcast_lcol(i, root_col, lrreq)
				j += 1
			else:
				if k == self.nb-1: break
				self.bcast_lcol(i, root_col, lrreq)
				self.bcast_rrow(j, root_row, lrreq)
			if next_root_row==self.row and next_root_col==self.col and (self.nprow-shrow) > 2 and do_diag_precompute:
				diag_precomputed = True

				yield from self.gemm((self.nprow-shrow)//2, self.npcol-shcol)
				shrow += (self.nprow-shrow)//2
				if pivreq: yield from self.wait_all("DIAG_BCAST", pivreq)
				if lrreq: yield from self.wait_all("WAIT", lrreq)

				yield from self.gemm(1, 1)
				yield from self.lazy_init(1, 1)
				yield from self.diag_trf()
				if k != self.nb-1:
					self.bcast_diag(next_root_row, next_root_col, pivreq)

				yield from self.gemm(self.nprow-shrow, self.npcol-shcol)
				shrow = i
				shcol = j
			else:
				diag_precomputed = False
				if do_lookahead_bcast and k < self.nb-2:
					if next_root_row == self.row and next_root_col == self.col:
						pass
					elif next_root_row == self.row:
						p = self.bcast_r(next_root_col)
					elif next_root_col  == self.col:
						p = self.bcast_l(next_root_row)
				yield from self.gemm(self.nprow-shrow, self.npcol-shcol)
				if pivreq: yield from self.wait_all("DIAG_BCAST", pivreq)
				if lrreq: yield from self.wait_all("WAIT", lrreq)
				shrow = i
				shcol = j
		self.log("TOTAL",0,self.env.now*1e-9,0)


if __name__ == '__main__':
	nrow = 4
	ncol = 3
	b = 288
	n = b* 286 * int(math.sqrt(nrow*ncol))
	nb = n // b
	env = simpy.Environment()
	hstores = []
	vstores = []
	for i in range(nrow):
		hstores.append([simpy.Store(env, capacity=1) for j in range(ncol)])
	for i in range(ncol):
		vstores.append([simpy.Store(env, capacity=1) for j in range(nrow)])
	procs = []
	for j in range(ncol):
		for i in range(nrow):
			nprow = (nb - i + nrow - 1) // nrow
			npcol = (nb - j + ncol - 1) // ncol
			procs.append(ProcComp(env, i, j, nrow, ncol, vstores[j], hstores[i], nb, nprow, npcol, b, True))
	env.run()
	for p in procs:
		p.print_log()
