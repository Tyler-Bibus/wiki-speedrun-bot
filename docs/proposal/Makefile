.PHONY: all

all: miniproject-proposal.pdf

# The first pass writes reference labels; the second resolves citations.
miniproject-proposal.pdf: miniproject-proposal.tex Makefile
	pdflatex -interaction=nonstopmode -halt-on-error miniproject-proposal.tex
	pdflatex -interaction=nonstopmode -halt-on-error miniproject-proposal.tex
