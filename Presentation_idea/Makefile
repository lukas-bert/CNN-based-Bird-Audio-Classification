all: build/main.pdf

texoptions = \
	     --lualatex \
	     --interaction=nonstopmode \
	     --halt-on-error \
	     --output-directory=build

build/main.pdf: FORCE | build
	latexmk $(texoptions) main.tex

preview: FORCE | build
	latexmk $(texoptions) -pvc main.tex

FORCE:

build:
	mkdir -p build

clean:
	rm -r build
