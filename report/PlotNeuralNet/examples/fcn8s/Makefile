all: build/fcn8.pdf

texoptions = \
	     --pdflatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/fcn8.pdf: FORCE | build
	latexmk $(texoptions) fcn8.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc fcn8.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
