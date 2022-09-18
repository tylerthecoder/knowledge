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
	with open("./data/" + filename + "_data.csv", 'w') as data:
		with open("./data/" + filename + "_labels.csv", 'w') as labels:
			for i in range(size):
				dp = gen_data_point()
				label = getLabel(dp)
				data.write(f'{dp[0]},{dp[1]}\n')
				labels.write(f'{label}\n')


n_training = 100000
n_validation = 20000
n_testing = 20000

# Generate training data
gen_data_and_write('training', n_training)
gen_data_and_write('validation', n_validation)
gen_data_and_write('testing', n_testing)
