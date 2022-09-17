import random

# Returns random point between -5 and 5
def gen_data_point():
	x = random.uniform(-5, 5)
	y = random.uniform(-5, 5)
	return (x, y)

def getLabel(data: tuple):
	x, y = data
	return 1 if x**2 + y**2 <= 2.5**2 else 0


def gen_data_and_write(filename: str, size: int):
	with open(filename, 'w') as f:
		for i in range(size):
			data = gen_data_point()
			label = getLabel(data)
			f.write(f'{data[0]},{data[1]},{label}\n')


n_training = 100000
n_validation = 20000
n_testing = 20000

# Generate training data
gen_data_and_write('./data/training_data.txt', n_training)
gen_data_and_write('./data/validation_data.txt', n_validation)
gen_data_and_write('./data/testing_data.txt', n_testing)
