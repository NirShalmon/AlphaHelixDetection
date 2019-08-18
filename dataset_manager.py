import mrcfile

def read_mrc(path):
	file = mrcfile.open(path)
	data = file.data
	file.close()
	return data


def dimentions(map):
	'''
	Returns a tuple of the size of a map
	'''
	return (len(map),len(map[0]),len(map[0][0]))


def check_same_size(a,b):
	return dimentions(a) == dimentions(b)


def apply_cutoff(map, cutoff):
	size = dimentions(map)
	for i in range(size[0]):
		for j in range(size[1]):
			for k in range(size[2]):
				if map[i][j][k] >= cutoff:
					map[i][j][k] = 1
				else:
					map[i][j][k] = 0


map = read_mrc('map.mrc')
helix_map = read_mrc('helix.mrc')
assert check_same_size(map,helix_map)
apply_cutoff(helix_map, 0.25)