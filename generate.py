import random
import numpy as np


class Piece:
	def __init__(self, blocks):
		self.blocks = blocks
		self.trim()
		if self.width > 8 or self.height > 8:
			raise RuntimeError('Piece too big')
		# TODO check that pieces are contiguous (mark and sweep)

	@property
	def width(self):
		return self.blocks.shape[0]

	@property
	def height(self):
		return self.blocks.shape[1]

	def trim(self):
		if not np.sum(self.blocks):
			return 
		# cut off any zero borders
		while not np.sum(self.blocks[0,:]):
			self.blocks = self.blocks[1:,:]
		while not np.sum(self.blocks[-1,:]):
			self.blocks = self.blocks[:-1,:]
		while not np.sum(self.blocks[:,0]):
			self.blocks = self.blocks[:,1:]
		while not np.sum(self.blocks[:,-1]):
			self.blocks = self.blocks[:,:-1]

	def hash(self):
		box = np.zeros((8,8), dtype=np.int8)
		box[:self.width, :self.height] = self.blocks
		hsh = 0
		for blk in box.flat:
			hsh = (hsh << 1) | (1 if blk else 0)
		return hsh

	def rotated(self):
		return Piece(np.rot90(self.blocks))

	def flipped(self):
		return Piece(np.flipud(self.blocks))
	
	def permutations(self):
		piece = Piece(np.copy(self.blocks))
		perms = {}
		for parity in range(2):
			for rotation in range(4):
				piece = piece.rotated()
				perms[piece.hash()] = piece
			piece = piece.flipped()
		return perms.values()

	def piece_masks(self, field):
		for piece in self.permutations():
			for x in range(field.width - piece.width + 1):
				for y in range(field.height - piece.height + 1):
					piece_mask = np.zeros(field.blocks.shape, dtype=np.int8)
					piece_mask[x:x+piece.width, y:y+piece.height] = piece.blocks
					yield piece_mask


class Field:
	def __init__(self, width, height):
		self.blocks = np.zeros((width, height), dtype=np.int8)

	@property
	def width(self):
		return self.blocks.shape[0]

	@property
	def height(self):
		return self.blocks.shape[1]

	def __repr__(self):
		block_w = 5
		block_h = 3
		out_w = (block_w + 1) * self.width - 1
		out_h = (block_h + 1) * self.height - 1

		out_s = ''
		for row in range(out_h):
			logical_h, inner_h = divmod(row, (block_h+1))
			is_h_padding = inner_h == block_h
			for col in range(out_w):
				logical_w, inner_w = divmod(col, (block_w+1))
				is_w_padding = inner_w == block_w
				this_block = self.blocks[logical_w, logical_h]

				if not is_h_padding and not is_w_padding:
					out_s += str(this_block)
				elif is_w_padding and not is_h_padding:
					connect = (
						logical_w +1 < self.width and
						this_block == self.blocks[logical_w+1, logical_h]
					)
					out_s += str(this_block) if connect else ' '
				elif is_h_padding and not is_w_padding:
					connect = (
						logical_h +1 < self.height and
						this_block == self.blocks[logical_w, logical_h+1]
					)
					out_s += str(this_block) if connect else ' '
				elif is_h_padding and is_w_padding:
					connect = (
						logical_w +1 < self.width and
						logical_h +1 < self.height and
						this_block == self.blocks[logical_w, logical_h+1] and
						this_block == self.blocks[logical_w+1, logical_h] and
						this_block == self.blocks[logical_w+1, logical_h+1]
					)
					out_s += str(this_block) if connect else ' '
				else:
					out_s += ' '

			out_s += '\n'
		return self.render_framed(out_s)

	def render_framed(self, str):
		lines = str.split('\n')
		width, height = len(lines[0]), len(lines)
		framed = ' -' + '-' * (width + 2) + '- \n'
		framed += '| ' + ' ' * (width + 2) + ' |\n'
		for line in lines:
			if line:
				framed += f'|  {line}  |\n'
		framed += '| ' + ' ' * (width + 2) + ' |\n'
		framed += ' -' + '-' * (width + 2) + '- \n'
		return framed

	def put_piece_xy(self, piece, x, y):
		self.blocks[x:x+piece.width, y:y+piece.height] += piece.blocks

	def take_piece_xy(self, piece, x, y):
		self.blocks[x:x+piece.width, y:y+piece.height] -= piece.blocks

	def piece_fits_xy(self, piece, x, y):
		region = self.blocks[x:x+piece.width, y:y+piece.height]
		if region.shape != piece.blocks.shape:
			return False
		return not np.amax(np.bitwise_and(region, piece.blocks))

	def put_pm(self, piece_mask):
		self.blocks += piece_mask

	def take_pm(self, piece_mask):
		self.blocks -= piece_mask

	def pm_fits(self, piece_mask):
		return not np.amax(self.blocks * piece_mask)

	def free_block_exists(self):
		return np.amin(self.blocks) == 0

	def get_free_block_xy(self):
		if not self.free_block_exists():
			raise RuntimeError('No free block')
		while True:
			x = random.randint(0, self.width-1)
			y = random.randint(0, self.height-1)
			if not self.blocks[x, y]:
				return x, y

	def generate_puzzle(self, num_pieces):
		# position seeds (one block per piece)
		for piece in range(num_pieces):
			x, y = self.get_free_block_xy()
			self.blocks[x, y] = piece + 1
		while self.free_block_exists():
			x, y = self.get_free_block_xy()
			neighbor_blocks = []
			if x > 0 and self.blocks[x-1, y]:
				neighbor_blocks.append(self.blocks[x-1, y])
			if x < self.width-1 and self.blocks[x+1, y]:
				neighbor_blocks.append(self.blocks[x+1, y])
			if y > 0 and self.blocks[x, y-1]:
				neighbor_blocks.append(self.blocks[x, y-1])
			if y < self.height-1 and self.blocks[x, y+1]:
				neighbor_blocks.append(self.blocks[x, y+1])
			if neighbor_blocks:					
				self.blocks[x, y] = random.choice(neighbor_blocks)

	def grab_pieces(self):
		piece_numbers = list(set(np.unique(self.blocks)) - set([0]))
		piece_numbers = sorted(piece_numbers, reverse=True)
		pieces = []
		for piece_number in piece_numbers:
			piece_mask = np.floor_divide(self.blocks, piece_number) * piece_number
			self.take_pm(piece_mask)
			pieces.append(Piece(piece_mask))
		return pieces


class Puzzle:
	def __init__(self, field, pieces):
		self.field = field
		self.pieces = pieces
		self.num_solutions = 0

	def find_solutions(self):
		self.put_pieces(self.pieces)

	def put_pieces(self, pieces):
		if not pieces:
			self.num_solutions += 1
			print(f'Solution #{self.num_solutions}: \n{self.field}\n')
			return

		for pm in pieces[0].piece_masks(self.field):
			if self.field.pm_fits(pm):
				self.field.put_pm(pm)
				self.put_pieces(pieces[1:])
				self.field.take_pm(pm)


if __name__ == '__main__':
	size = 6
	field = Field(size, size)
	field.generate_puzzle(num_pieces=size)
	print(field)
	pieces = field.grab_pieces()

	puzzle = Puzzle(field, pieces)
	puzzle.find_solutions()
