import os
import random
from typing import List
from PIL import Image


# Paths
gan_dog_input_paths = ["data/dog_breeds/test/German Sheperd", "data/dog_breeds/train/German Sheperd", "data/dog_breeds/valid/German Sheperd"]
gan_dog_dir = "dogs2wolf/trainA"
gan_wolf_dir = "dogs2wolf/trainB"

def get_animal_input_path(animal: str) -> str:
	return "data/wild_animals/" + animal + "-resize-224/" + "resize-224"

def get_animal_test_output_path(animal: str) -> str:
	return "wild-animals/test/" + animal

def get_animal_train_output_path(animal: str) -> str:
	return "wild-animals/train/" + animal

# make the directories
os.makedirs("wild-animals/test", exist_ok=True)
os.makedirs("wild-animals/train", exist_ok=True)
os.makedirs(gan_dog_dir, exist_ok=True)
os.makedirs(gan_wolf_dir, exist_ok=True)

# Set animals
animals = ['cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']

test_animal = 'wolf'

percent_train = 0.9

percent_wolf_removed = 0.5

for animal in animals:
	animal_dir = get_animal_input_path(animal)

	print("Reading " + animal_dir)

	os.makedirs(get_animal_test_output_path(animal), exist_ok=True)
	os.makedirs(get_animal_train_output_path(animal), exist_ok=True)

	# Get all the files in the directory
	files = os.listdir(animal_dir)

	# Shuffle the files
	random.shuffle(files)

	# Remove random elements from the list
	num_train = int(len(files) * percent_train)

	print("Number of training images for " + animal + ": " + str(num_train))

	# Move the training files to the training directory
	for file in files[:num_train]:
		os.rename(animal_dir + "/" + file, get_animal_train_output_path(animal) + "/" + file)

	# Move the testing files to the testing directory
	for file in files[num_train:]:
		os.rename(animal_dir + "/" + file, get_animal_test_output_path(animal) + "/" + file)

	# Remove a percent of the wolf images from the training set
	num_wolf_to_remove = int(num_train * percent_wolf_removed)


	if animal == test_animal:
		print("Removing " + str(num_wolf_to_remove) + " " + animal + " images from the training set")

		# Move the wolf files to the GAN directory
		for file in files[:num_wolf_to_remove]:
			os.rename(get_animal_train_output_path(animal) + "/" + file, gan_wolf_dir + "/" + file)


# Move the dog files to the GAN directory

file_paths: List[str] = []

for path in gan_dog_input_paths:
	file_paths.extend([path + "/" + file for file in os.listdir(path)])

print("Moving " + str(len(file_paths)) + " dog images to the GAN directory")

for index, file in enumerate(file_paths):
	os.rename(file, gan_dog_dir + "/" + str(index) + ".jpg")


# Convert the wolf images from png to jpeg
for index, file in enumerate(os.listdir(gan_wolf_dir)):
	im = Image.open(gan_wolf_dir + "/" + file)
	rgb_im = im.convert('RGB')
	rgb_im.save(gan_wolf_dir + "/" + str(index) + ".jpg")
	rgb_im.close()
	os.remove(gan_wolf_dir + "/" + file)

