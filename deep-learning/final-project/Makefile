
unzip:
	unzip data.zip -d data

format:
	mv data/wild_animals/fox-resize-224/fox-resize-224 data/wild_animals/fox-resize-224/resize-224 || true
	mv data/wild_animals/lion-resize-224/lion-resize-224 data/wild_animals/lion-resize-224/resize-224 || true
	mv data/wild_animals/tiger-resize-224/tiger-resize-224 data/wild_animals/tiger-resize-224/resize-224 || true
	mv data/wild_animals/wolf-resize-224/wolf-resize-224 data/wild_animals/wolf-resize-224/resize-224 || true

clean:
	rm -rf data
	rm -rf dogs2wolf
	rm -rf wild-animals

run:
	python format.py

package:
	zip -r cycle-gan.zip Makefile README.md format.py