all: build/alexnet.pdf

texoptions = \
	     --pdflatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/alexnet.pdf: FORCE | build
	latexmk $(texoptions) alexnet.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc alexnet.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
