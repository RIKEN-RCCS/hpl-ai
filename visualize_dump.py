import sys
import glob
import array
import struct
import numpy as np
import moderngl as mgl
import moderngl_window as mglw

# random palette
palette = [
	'0x00d000', # COMM 1
	'0x10c030', # COMM 2
	'0x30c010', # COMM 3
	'0xd00000', # WAIT
	'0x3010c0', # COMP 1
	'0x1030c0', # COMP 2
	'0x2020c0', # COMP 3
	'0x4000b0', # COMP 4
	'0x0000d0', # COMP 5
	'0x0040b0', # COMP 6
	'0xcccccc', # MISC
	'0x0000a0', # COMP 7
	]
def hex2tup(pal):
	string = palette[pal]
	return (int(string[2:4],16), int(string[4:6],16), int(string[6:8],16))
tag2pal = {'DIAG_BCAST':0, 'LCOL_BCAST':1, 'RROW_BCAST':2, 'WAIT':3, 'TEST':18, 'DIAG_LU':4, 'TRSM_L':5, 'TRSM_R':6, 'CONV_L':7, 'CONV_R':8,
	'GEMM_UPDATE':9, 'LAZY_INIT':10, 'MISC':11, 'TOTAL':12, 'IR_GEMV':13, 'IR_GEMV_COMM':14, 'IR_TRSV':15, 'IR_TRSV_MV':16, 'IR_TRSV_COMM':17,
	'IT_REF_GEMV':13, 'IT_REF_TRSV':15, 'GEMM_PROGRESS':19}
pal2tag = dict((v, k) for k, v in tag2pal.items())
paletteu = [
	hex2tup(0),
	hex2tup(1),
	hex2tup(2),
	hex2tup(3),
	hex2tup(4),
	hex2tup(5),
	hex2tup(6),
	hex2tup(7),
	hex2tup(7),
	hex2tup(8),
	hex2tup(9),
	hex2tup(10),
	hex2tup(10),
	hex2tup(8),
	hex2tup(1),
	hex2tup(4),
	hex2tup(5),
	hex2tup(0),
	hex2tup(3),
	hex2tup(11),
	]

# converted from mplus_h10r
# see the license https://mplus-fonts.osdn.jp/mplus-bitmap-fonts/index.html#license
font_size = (7, 13, 0, -2)
font = {
	' ':((5,0,0,0,0),[]),
	'!':((4,1,7,1,0),[0x00,0x01,0x02,0x03,0x04,0x06,]),
	'"':((6,3,3,1,5),[0x00,0x20,0x01,0x21,0x02,0x22,]),
	'#':((6,5,7,0,0),[0x10,0x30,0x11,0x31,0x02,0x12,0x22,0x32,0x42,0x13,0x33,0x04,0x14,0x24,0x34,0x44,0x15,0x35,0x16,0x36,]),
	'$':((6,5,8,0,-1),[0x20,0x11,0x21,0x31,0x41,0x02,0x22,0x13,0x23,0x33,0x24,0x44,0x25,0x45,0x06,0x16,0x26,0x36,0x27,]),
	'%':((6,5,7,0,0),[0x00,0x10,0x01,0x11,0x41,0x32,0x23,0x14,0x05,0x35,0x45,0x36,0x46,]),
	'&':((6,5,7,0,0),[0x10,0x20,0x01,0x31,0x02,0x32,0x13,0x23,0x43,0x04,0x34,0x05,0x35,0x16,0x26,0x46,]),
	'\'':((4,1,3,1,5),[0x00,0x01,0x02,]),
	'(':((6,3,9,1,-1),[0x20,0x11,0x12,0x03,0x04,0x05,0x16,0x17,0x28,]),
	')':((6,3,9,1,-1),[0x00,0x11,0x12,0x23,0x24,0x25,0x16,0x17,0x08,]),
	'*':((6,5,6,0,1),[0x20,0x01,0x21,0x41,0x12,0x22,0x32,0x13,0x23,0x33,0x04,0x24,0x44,0x25,]),
	'+':((6,5,5,0,1),[0x20,0x21,0x02,0x12,0x22,0x32,0x42,0x23,0x24,]),
	',':((6,2,2,1,-1),[0x10,0x01,]),
	'-':((6,5,1,0,3),[0x00,0x10,0x20,0x30,0x40,]),
	'.':((6,1,1,2,0),[0x00,]),
	'/':((7,4,8,1,-1),[0x30,0x31,0x22,0x23,0x14,0x15,0x06,0x07,]),
	'0':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'1':((6,3,7,0,0),[0x20,0x11,0x21,0x02,0x22,0x23,0x24,0x25,0x26,]),
	'2':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x42,0x23,0x33,0x14,0x05,0x06,0x16,0x26,0x36,0x46,]),
	'3':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x31,0x22,0x13,0x23,0x33,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'4':((6,5,7,0,0),[0x20,0x30,0x11,0x31,0x02,0x32,0x03,0x33,0x04,0x14,0x24,0x34,0x44,0x35,0x36,]),
	'5':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x01,0x02,0x12,0x22,0x32,0x43,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'6':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x02,0x12,0x22,0x32,0x03,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'7':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x41,0x32,0x23,0x24,0x15,0x16,]),
	'8':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x13,0x23,0x33,0x04,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'9':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x43,0x14,0x24,0x34,0x44,0x45,0x16,0x26,0x36,]),
	':':((6,1,4,2,1),[0x00,0x03,]),
	';':((6,2,5,1,0),[0x10,0x13,0x04,]),
	'<':((7,4,7,1,0),[0x30,0x21,0x12,0x03,0x14,0x25,0x36,]),
	'=':((6,5,3,0,2),[0x00,0x10,0x20,0x30,0x40,0x02,0x12,0x22,0x32,0x42,]),
	'>':((7,4,7,1,0),[0x00,0x11,0x22,0x33,0x24,0x15,0x06,]),
	'?':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x33,0x24,0x26,]),
	'@':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x22,0x32,0x42,0x03,0x23,0x43,0x04,0x24,0x34,0x44,0x05,0x16,0x26,0x36,]),
	'A':((6,5,7,0,0),[0x20,0x21,0x12,0x32,0x13,0x33,0x04,0x14,0x24,0x34,0x44,0x05,0x45,0x06,0x46,]),
	'B':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x13,0x23,0x33,0x04,0x44,0x05,0x45,0x06,0x16,0x26,0x36,]),
	'C':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x03,0x04,0x05,0x45,0x16,0x26,0x36,]),
	'D':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x43,0x04,0x44,0x05,0x45,0x06,0x16,0x26,0x36,]),
	'E':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x01,0x02,0x03,0x13,0x23,0x33,0x04,0x05,0x06,0x16,0x26,0x36,0x46,]),
	'F':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x01,0x02,0x03,0x13,0x23,0x33,0x04,0x05,0x06,]),
	'G':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x03,0x23,0x33,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,0x46,]),
	'H':((6,5,7,0,0),[0x00,0x40,0x01,0x41,0x02,0x42,0x03,0x13,0x23,0x33,0x43,0x04,0x44,0x05,0x45,0x06,0x46,]),
	'I':((4,1,7,1,0),[0x00,0x01,0x02,0x03,0x04,0x05,0x06,]),
	'J':((6,5,7,0,0),[0x40,0x41,0x42,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'K':((6,5,7,0,0),[0x00,0x40,0x01,0x31,0x02,0x22,0x03,0x13,0x04,0x24,0x05,0x35,0x06,0x46,]),
	'L':((6,5,7,0,0),[0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x16,0x26,0x36,0x46,]),
	'M':((8,7,7,0,0),[0x00,0x60,0x01,0x11,0x51,0x61,0x02,0x22,0x42,0x62,0x03,0x33,0x63,0x04,0x64,0x05,0x65,0x06,0x66,]),
	'N':((6,5,7,0,0),[0x00,0x40,0x01,0x11,0x41,0x02,0x22,0x42,0x03,0x33,0x43,0x04,0x44,0x05,0x45,0x06,0x46,]),
	'O':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'P':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x13,0x23,0x33,0x04,0x05,0x06,]),
	'Q':((6,6,9,0,-2),[0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,0x37,0x48,0x58,]),
	'R':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x01,0x41,0x02,0x42,0x03,0x13,0x23,0x33,0x04,0x44,0x05,0x45,0x06,0x46,]),
	'S':((6,5,7,0,0),[0x10,0x20,0x30,0x01,0x41,0x02,0x13,0x23,0x33,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'T':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x21,0x22,0x23,0x24,0x25,0x26,]),
	'U':((6,5,7,0,0),[0x00,0x40,0x01,0x41,0x02,0x42,0x03,0x43,0x04,0x44,0x05,0x45,0x16,0x26,0x36,]),
	'V':((6,5,7,0,0),[0x00,0x40,0x01,0x41,0x02,0x42,0x13,0x33,0x14,0x34,0x25,0x26,]),
	'W':((10,9,7,0,0),[0x00,0x40,0x80,0x01,0x41,0x81,0x12,0x42,0x72,0x13,0x33,0x53,0x73,0x14,0x34,0x54,0x74,0x25,0x65,0x26,0x66,]),
	'X':((6,5,7,0,0),[0x00,0x40,0x11,0x31,0x22,0x23,0x24,0x15,0x35,0x06,0x46,]),
	'Y':((6,5,7,0,0),[0x00,0x40,0x01,0x41,0x12,0x32,0x13,0x33,0x24,0x25,0x26,]),
	'Z':((6,5,7,0,0),[0x00,0x10,0x20,0x30,0x40,0x41,0x32,0x23,0x14,0x05,0x06,0x16,0x26,0x36,0x46,]),
	'[':((6,3,9,1,-1),[0x00,0x10,0x20,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x18,0x28,]),
	'\\':((7,4,8,1,-1),[0x00,0x01,0x12,0x13,0x24,0x25,0x36,0x37,]),
	']':((6,3,9,1,-1),[0x00,0x10,0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x08,0x18,0x28,]),
	'^':((6,5,3,0,5),[0x20,0x11,0x31,0x02,0x42,]),
	'_':((6,5,1,0,-1),[0x00,0x10,0x20,0x30,0x40,]),
	'`':((5,2,4,1,4),[0x00,0x01,0x12,0x13,]),
	'a':((5,4,5,0,0),[0x10,0x20,0x31,0x12,0x22,0x32,0x03,0x33,0x14,0x24,0x34,]),
	'b':((5,4,7,0,0),[0x00,0x01,0x02,0x12,0x22,0x03,0x33,0x04,0x34,0x05,0x35,0x06,0x16,0x26,]),
	'c':((5,4,5,0,0),[0x10,0x20,0x01,0x31,0x02,0x03,0x14,0x24,0x34,]),
	'd':((5,4,7,0,0),[0x30,0x31,0x12,0x22,0x32,0x03,0x33,0x04,0x34,0x05,0x35,0x16,0x26,0x36,]),
	'e':((5,4,5,0,0),[0x10,0x20,0x01,0x31,0x02,0x12,0x22,0x32,0x03,0x14,0x24,0x34,]),
	'f':((5,4,7,0,0),[0x20,0x30,0x11,0x12,0x03,0x13,0x23,0x33,0x14,0x15,0x16,]),
	'g':((5,4,7,0,-2),[0x10,0x20,0x30,0x01,0x31,0x02,0x32,0x03,0x33,0x14,0x24,0x34,0x35,0x16,0x26,]),
	'h':((5,4,7,0,0),[0x00,0x01,0x02,0x12,0x22,0x03,0x33,0x04,0x34,0x05,0x35,0x06,0x36,]),
	'i':((4,1,8,1,0),[0x00,0x03,0x04,0x05,0x06,0x07,]),
	'j':((5,3,10,0,-2),[0x20,0x23,0x24,0x25,0x26,0x27,0x28,0x09,0x19,]),
	'k':((5,4,7,0,0),[0x00,0x01,0x02,0x32,0x03,0x23,0x04,0x14,0x05,0x25,0x06,0x36,]),
	'l':((4,1,7,1,0),[0x00,0x01,0x02,0x03,0x04,0x05,0x06,]),
	'm':((6,5,5,0,0),[0x00,0x10,0x20,0x30,0x01,0x21,0x41,0x02,0x22,0x42,0x03,0x23,0x43,0x04,0x24,0x44,]),
	'n':((5,4,5,0,0),[0x00,0x10,0x20,0x01,0x31,0x02,0x32,0x03,0x33,0x04,0x34,]),
	'o':((5,4,5,0,0),[0x10,0x20,0x01,0x31,0x02,0x32,0x03,0x33,0x14,0x24,]),
	'p':((5,4,7,0,-2),[0x00,0x10,0x20,0x01,0x31,0x02,0x32,0x03,0x33,0x04,0x14,0x24,0x05,0x06,]),
	'q':((5,4,7,0,-2),[0x10,0x20,0x30,0x01,0x31,0x02,0x32,0x03,0x33,0x14,0x24,0x34,0x35,0x36,]),
	'r':((5,4,5,0,0),[0x00,0x20,0x30,0x01,0x11,0x02,0x03,0x04,]),
	's':((5,4,5,0,0),[0x10,0x20,0x30,0x01,0x12,0x22,0x33,0x04,0x14,0x24,]),
	't':((5,4,7,0,0),[0x10,0x11,0x02,0x12,0x22,0x32,0x13,0x14,0x15,0x26,0x36,]),
	'u':((5,4,5,0,0),[0x00,0x30,0x01,0x31,0x02,0x32,0x03,0x33,0x14,0x24,0x34,]),
	'v':((6,5,5,0,0),[0x00,0x40,0x01,0x41,0x12,0x32,0x13,0x33,0x24,]),
	'w':((6,5,5,0,0),[0x00,0x20,0x40,0x01,0x21,0x41,0x02,0x22,0x42,0x13,0x33,0x14,0x34,]),
	'x':((5,4,5,0,0),[0x00,0x30,0x01,0x31,0x12,0x22,0x03,0x33,0x04,0x34,]),
	'y':((5,4,7,0,-2),[0x00,0x30,0x01,0x31,0x02,0x32,0x03,0x33,0x14,0x24,0x34,0x35,0x16,0x26,]),
	'z':((5,4,5,0,0),[0x00,0x10,0x20,0x30,0x21,0x12,0x03,0x04,0x14,0x24,0x34,]),
	'{':((8,5,9,1,-1),[0x30,0x40,0x21,0x22,0x23,0x04,0x14,0x25,0x26,0x27,0x38,0x48,]),
	'|':((6,1,9,2,-1),[0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,]),
	'}':((8,5,9,1,-1),[0x00,0x10,0x21,0x22,0x23,0x34,0x44,0x25,0x26,0x27,0x08,0x18,]),
	'~':((6,5,2,0,5),[0x10,0x20,0x40,0x01,0x21,0x31,]), }

def onetwofiveseq(x):
	base=1
	while True:
		if x <= 10*base:
			return base
		if x <= 20*base:
			return 2*base
		if x <= 50*base:
			return 5*base
		base *= 10
itrees = []
class GLWindow(mglw.WindowConfig):
	gl_version = (3, 3)
	window_size = (400, 300)
	aspect_ratio = None
	title = "dump viewer"
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.itrees = itrees
		self.tbegin = 0.
		self.tend = 1.
		self.nprocs = itrees.len()
		self.idbegin = 0.
		self.idend = min(1.*self.nprocs, 16.)
		#for i in range(min(self.nprocs, 16)):
		#	self.itrees[i]
		self.modifier = None
		self.text_info = ""
		self.create_window()
		self.last_intbegin = -1
		self.last_intend = -1
		self.last_id = -1
		self.last_pos = -1
		self.program()
		self.globjects()
		self.gen_fonts()
		
	def program(self):
		self.single_color= self.ctx.program(vertex_shader='''
#version 330
in vec2 vert;
uniform vec2 xy_start;
uniform vec2 xy_scale;
void main() {
	gl_Position = vec4(xy_scale*(vert-xy_start), 0., 1.);
}
''',
			fragment_shader='''
#version 330
out vec4 fcolor;
uniform vec4 icolor;
void main() {
	fcolor = icolor;
}
''')
		self.sc_xy_start = self.single_color['xy_start']
		self.sc_xy_scale = self.single_color['xy_scale']
		self.sc_color = self.single_color['icolor']
		self.prog = self.ctx.program(vertex_shader='''
#version 330
in vec2 vert2;
in uvec3 icolor2;
out vec3 frag_color;
uniform vec2 xy_start2;
uniform vec2 xy_scale2;
void main() {
	frag_color = icolor2/255.;
	gl_Position = vec4(xy_scale2*(vert2-xy_start2), 0., 1.);
}
''',
			fragment_shader='''
#version 330
in vec3 frag_color;
out vec4 color;
void main() {
	color = vec4(frag_color, 1.);
}
''')
		self.xy_start = self.prog['xy_start2']
		self.xy_scale = self.prog['xy_scale2']
		self.font_prog = self.ctx.program(vertex_shader='''
#version 330
in uvec2 vert;
uniform vec2 xy_start;
uniform vec2 xy_scale;
void main() {
	gl_Position = vec4(xy_scale*(vert-xy_start), 0., 1.);
}
''',
			fragment_shader='''
#version 330
out vec4 fcolor;
uniform vec4 icolor;
void main() {
	fcolor = icolor;
}
''')
		self.font_xy_start = self.font_prog['xy_start']
		self.font_xy_scale = self.font_prog['xy_scale']
		self.font_color = self.font_prog['icolor']
	def globjects(self):
		self.xlabels_lines_vbo = self.ctx.buffer(reserve=8*(self.width+1), dynamic=True)
		self.xlabels_lines_vao = self.ctx.simple_vertex_array(self.single_color, self.xlabels_lines_vbo, 'vert')
		self.data_vers = [None for _ in range(self.nprocs)]
		self.data_cols = [None for _ in range(self.nprocs)]
		self.data_vaos = [None for _ in range(self.nprocs)]

	def gen_fonts(self):
		self.font_vbo = {}
		self.font_vao = {}
		for k, v in font.items():
			vert = array.array('b')
			shift = v[0]
			for d in v[1]:
				y = (shift[2] - d%16) +shift[4]
				x = d//16 + shift[3]
				vert.append(x)
				vert.append(y)
			if len(vert)==0: continue
			self.font_vbo[k] = self.ctx.buffer(vert.tobytes())
			self.font_vao[k] = self.ctx.vertex_array(self.font_prog, [(self.font_vbo[k], '2i1 /v', 'vert')])


	def create_window(self):
		self.width = 400
		self.height = 300
	
	def key_event(self, key, action, modifiers):
		keys = self.wnd.keys

		if action == keys.ACTION_PRESS:
			if key == keys.Q:
				sys.exit()
				# XXX
			elif key == keys.Z:
				self.modifier = 1
			elif key == keys.W or key == keys.UP:
				self.idbegin -= 1
				self.idend -= 1
			elif key == keys.S or key == keys.DOWN:
				self.idbegin += 1
				self.idend += 1
			elif key == keys.A or key == keys.LEFT:
				self.on_shiftleft()
			elif key == keys.D or key == keys.RIGHT:
				self.on_shiftright()
			elif key == keys.O:
				self.on_scaleup(self.width/2, self.height/2)
			elif key == keys.I:
				self.on_scaledown(self.width/2, self.height/2)
		elif action == keys.ACTION_RELEASE:
			if key == keys.Z:
				self.modifier = 0


	def mouse_position_event(self, x, y, dx, dy):
		self.mouse_x = x
		self.mouse_y = y
		self.text_info = self.get_pos_info(x, y)

	def mouse_drag_event(self, x, y, dx, dy):
		self.shift_xy(dx, dy)
	def mouse_scroll_event(self, x_offset, y_offset):
		if y_offset > 0:
			if self.modifier:
				self.on_scaledown(self.mouse_x, self.mouse_y)
			else:
				self.on_shiftright()
		elif y_offset < 0:
			if self.modifier:
				self.on_scaleup(self.mouse_x, self.mouse_y)
			else:
				self.on_shiftleft()


	def on_scaleup(self, x, y):
		if x < 0 or x > self.width:
			return
		x0 = (x - 0) / (self.width-0) * (self.tend - self.tbegin) + self.tbegin
		tdeltab = self.tbegin - x0
		tdeltae = self.tend - x0
		self.tbegin = x0 + tdeltab*1.1
		self.tend = x0 + tdeltae*1.1

	def on_scaledown(self, x, y):
		if x < 0 or x > self.width:
			return
		x0 = (x - 0) / (self.width-0) * (self.tend - self.tbegin) + self.tbegin
		tdeltab = self.tbegin - x0
		tdeltae = self.tend - x0
		self.tbegin = x0 + tdeltab/1.1
		self.tend = x0 + tdeltae/1.1
	def on_shiftright(self):
		tdelta = self.tend - self.tbegin
		self.tbegin = self.tbegin + tdelta*0.1
		self.tend = self.tend + tdelta*0.1

	def on_shiftleft(self):
		tdelta = self.tend - self.tbegin
		self.tbegin = self.tbegin - tdelta*0.1
		self.tend = self.tend - tdelta*0.1

	def shift_xy(self, deltax, deltay):
		unitw = (self.width-0) / (self.tend - self.tbegin)
		deltax /= unitw
		unith = (self.height-0) / (self.idend - self.idbegin)
		deltay /= unith
		self.idbegin -= deltay
		self.idend -= deltay
		self.tbegin -= deltax
		self.tend -= deltax


	def get_pos_info(self, x, y):
		x0 = (x - 0) / (self.width-0) * (self.tend - self.tbegin) + self.tbegin
		y0 = (y) / (self.height-0) * (self.idend - self.idbegin) + self.idbegin
		id = int(y0)
		pos = int((y0-id-0.05)/0.225)
		if id < 0 or id >= self.nprocs:
			return
		info_text = ""
		for idx in self.itrees[id][0].find_overlap(x0):
			begin = self.itrees[id][1][idx]
			end = self.itrees[id][2][idx]
			tag = self.itrees[id][3][idx]
			mypos = self.itrees[id][4][idx]
			ops = self.itrees[id][5][idx]
			if begin <= x0 and x0 <= end and mypos == pos:
				info_text += "{:s} {:>.6f} -- {:>.6f} = {:>.6f}sec {:>.3f}gops".format(
					pal2tag[tag],
					round(begin,6), round(end,6), round((end-begin),6),
					round(1e-9*ops/(end-begin),6))
		return info_text


	def resize(self, width, height):
		self.width = width
		self.height = height

	def text_width(self, text):
		x = 0
		for ch in text:
			if not ch in font:
				continue
			shift = font[ch][0]
			x += shift[0]
		return x
	def draw_text(self, x0, y0, text):
		for ch in text:
			if not ch in font:
				continue
			shift = font[ch][0]
			if ch in self.font_vao:
				self.font_xy_start.value = (-x0+self.width/2, -y0+self.height/2)
				self.font_vao[ch].render(mode=mgl.POINTS)
			x0 += shift[0]

	def gen_vertex_array(self, id):
		vers = []
		cols = []
		n=0
		sz = len(self.itrees[id][1])
		for idx in range(sz):
			x0 = self.itrees[id][1][idx]
			x1 = self.itrees[id][2][idx]
			tag = self.itrees[id][3][idx]
			pos = self.itrees[id][4][idx]
			y0 = id + 0.05 + pos*0.225
			y1 = id + 0.275 + pos*0.225
			col = paletteu[tag]
			vers.append((x0,y0))
			vers.append((x1,y0))
			vers.append((x0,y1))
			vers.append((x0,y1))
			vers.append((x1,y0))
			vers.append((x1,y1))
			cols.append((col[0], col[1], col[2]))
			cols.append((col[0], col[1], col[2]))
			cols.append((col[0], col[1], col[2]))
			cols.append((col[0], col[1], col[2]))
			cols.append((col[0], col[1], col[2]))
			cols.append((col[0], col[1], col[2]))
			n+=1
		vers = np.array(vers, dtype=np.float32)
		cols = np.array(cols, dtype='uint8')
		return (vers, cols, n)
	def draw_id(self, id):
		if self.data_vers[id] is None:
			v, c, s = self.gen_vertex_array(id)
			self.data_vers[id] = self.ctx.buffer(v.tobytes())
			self.data_cols[id] = self.ctx.buffer(c.tobytes())
			self.data_vaos[id] = self.ctx.vertex_array(self.prog,
				[(self.data_vers[id], '2f /v', 'vert2'),
				(self.data_cols[id], '3u1 /v', 'icolor2')])
		self.data_vaos[id].render()

		return

	def draw_xlabels(self):
		interval = (self.tend - self.tbegin)*1e9
		step = int(onetwofiveseq(interval+1))
		start = (int(self.tbegin*1e9)+step-1)//step*step

		yt = font_size[1] - font_size[3]
		y0 = yt + font_size[1]
		y1 = y0 + font_size[1]
		vert = array.array('f')
		vert.extend([0, y0, self.width, y0])
		for x in range(start,int(self.tend*1e9),step):
			x0 = (x*1e-9-self.tbegin) / (self.tend - self.tbegin) * self.width
			vert.extend([x0, y0, x0, y1])
			self.draw_text(x0, yt, str(round(x*1e-9,6)))
		
		self.xlabels_lines_vbo.orphan(size=len(vert)*4)
		self.xlabels_lines_vbo.write(vert.tobytes())
		self.xlabels_lines_vao.render(mode=mgl.LINES)
	def draw_focus_label(self):
		if self.text_info:
			width = self.text_width(self.text_info)
			x = self.width/2 - width/2
			x = max(x, 0)
			y = -font_size[3]
			self.draw_text(x, y, self.text_info)

	def draw_labels(self):
		pass

	def render(self, time, frametime):
		self.xy_start.value = ((self.tbegin+self.tend)/2, (self.idbegin+self.idend)/2)
		self.xy_scale.value = (2./(self.tend-self.tbegin), -2./(self.idend-self.idbegin))
		self.sc_xy_start.value = (self.width/2, self.height/2)
		self.sc_xy_scale.value = (2./self.width, 2./self.height)
		self.font_xy_scale.value = (2./self.width, 2./self.height)
		self.ctx.clear(0.7, 0.7, 0.7, 1.)
		for id in range(self.nprocs):
			if id >= self.idbegin-1 and id <= self.idend+1:
				self.draw_id(id)
		self.sc_color.value = (0., 0., 0., 1.)
		self.font_color.value = (0., 0., 0., 1.)
		self.draw_xlabels()
		self.draw_focus_label()
		#gl.glLoadIdentity()
		#self.set_viewport()
		#self.draw_xlabels()

class Intervals:
	# simple, small, stupid implementation of an interval tree structure
	# this is well suited for spatialy uniformly distributed, small segmented intervals.
	small_limit = 30
	def __init__(self, beg, end, idx, depth=0):
		if len(beg) < self.small_limit or depth > 20:
			self.small = None
			self.large = None
			self.idx = idx 
		else:
			min = np.min(beg)
			max = np.max(end)
			self.sep = (min + max)/2
			smaller = end < self.sep
			larger = beg > self.sep
			mid = np.logical_not(np.logical_or(smaller, larger))
			if np.any(smaller):
				self.small = Intervals(beg[smaller], end[smaller], idx[smaller], depth+1)
			else:
				self.small = None
			if np.any(larger):
				self.large = Intervals(beg[larger], end[larger], idx[larger], depth+1)
			else:
				self.large = None
			self.idx= idx[mid]
	def find_overlap(self, pos):
		child = np.array([], dtype='int64')
		if not self.small is None and pos < self.sep:
			child = self.small.find_overlap(pos)
		if not self.large is None and pos > self.sep:
			child = self.large.find_overlap(pos)
		return np.append(child, self.idx)

def readdata(filename):
	print("reading {}".format(filename))
	with open(filename, "rb") as f:
		d = {}
		for line in f:
			line = line.decode('utf-8')
			split = line.split(',')
			if split[0] == 'bio':
				split[2] = split[2].strip()
				num = int(split[1])
				begend = split[2][0:3]
				tag = split[2][4:]
				if begend != 'ACC':
					form = "@{}d".format(num)
				else:
					form = "@{}Q".format(num)
				size = struct.calcsize(form)
				bytes = f.read(size)
				ups = struct.unpack(form, bytes)
				if begend != 'PUT':
					if not tag in d:
						d[tag] = (array.array('d'), array.array('d'), array.array('Q'))
					if begend == 'BEG':
						for v in ups:
							d[tag][0].append(v)
					elif begend == 'END':
						for v in ups:
							d[tag][1].append(v)
					elif begend == 'ACC':
						for v in ups:
							d[tag][2].append(v)
				continue
			elif len(split) == 4:
				split[2] = split[2].strip()
				begend = split[2][0:3]
				if begend == 'PUT':
					continue
				tag = split[2][4:]
				time = float(split[1])
				if not tag in d:
					d[tag] = (array.array('d'), array.array('d'), array.array('Q'))
				if begend == 'BEG':
					d[tag][0].append(time)
				elif begend == 'END':
					d[tag][1].append(time)
				elif begend == 'ACC':
					d[tag][2].append(int(split[0]))
	return d
def analyzedata(filename, tag2pal):
	d = readdata(filename)
	sz = 0
	begs = array.array('d')
	ends = array.array('d')
	pals = array.array('B')
	poss = array.array('B')
	accs = array.array('Q')
	for tag, ditem in d.items():
		if tag in tag2pal:
			pal = tag2pal[tag]
		else:
			pal = tag2pal['MISC']
		size = len(ditem[0])
		if size != len(ditem[1]) or size != len(ditem[2]):
			print("file format error")
			sys.exit(1)
		begs.extend(ditem[0])
		ends.extend(ditem[1])
		accs.extend(ditem[2])
		sz += size
		for _ in range(size):
			pals.append(pal)
			poss.append(0)
	begs = np.asarray(begs, dtype='double')
	ends = np.asarray(ends, dtype='double')
	sidx = np.argsort(begs)
	stack = np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype='double')
	for x in sidx:
		for e, v in enumerate(stack):
			if v <= begs[x]:
				stack[e] = ends[x]
				poss[x] = e
				break
	tree = Intervals(begs, ends, np.arange(0,sz,1,dtype='int64'))
	return (tree, begs, ends, pals, poss, accs)
class LazyData:
	def __init__(self, filenames):
		self.nprocs = len(filenames)
		self.filenames = filenames
		self.evaluated = [None for _ in range(self.nprocs)]
	def __getitem__(self, idx):
		if self.evaluated[idx] is None:
			self.evaluated[idx] = analyzedata(self.filenames[idx], tag2pal)
		return self.evaluated[idx]#.result()
	def len(self):
		return self.nprocs


if __name__ == '__main__':
	print("python visualize_dump.py \'filename_with_wildcard.dump\'")
	print("You need quotes to glob correctly.")
	print("q to exit.")
	print("Drag to move around.")
	print("Wheel to scroll. Hold z and wheel to zoom.")

	itrees = LazyData(sorted(glob.glob(sys.argv[1])))
	del sys.argv[1]
	mglw.run_window_config(GLWindow)


