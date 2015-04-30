LATEX=latexmk
SRCS:=$(shell find . -name "*.tex")

DFFLAG=-xelatex -interaction=nonstopmode
RFLAG=-quiet
#DFLAG=-pv

debug:
	$(LATEX) $(DFFLAG) $(DFLAG)
release:
	$(LATEX) $(DFFLAG) $(RFLAG)
	$(LATEX) -c
clean:
	$(LATEX) -C
