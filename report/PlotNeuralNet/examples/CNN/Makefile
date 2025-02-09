all: build/cnn.pdf

texoptions = \
	     --pdflatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/cnn.pdf: FORCE | build
	latexmk $(texoptions) cnn.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc cnn.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
