## Run

create a `data` directory that looks like this

- data
  - dog_breeds
    - test
    - train
    - valid
  - wild_animals
    - cheetah-resize-224

run `make format` to clean up some directory name mismatches

run `make run` to convert the files

## Goal

Output file format

Classification

- wild-animals
  - train
    - cheetah
    - lion
    - ...
  - test
    - cheetah
    - lion
    - ...

test folder will contain random 10% of each class
Remove X% of the wolf training images (will use cycle gan to generate)

Cycle-Gan

- dogs2wolf
  - trainA (dog images)
  - trainB (wolf images)

The dog images will be the german shepard imgs
Wolf images come from removed images in classification

convert pngs to jpegs
