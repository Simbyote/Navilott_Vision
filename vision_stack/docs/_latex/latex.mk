# latex.mk --- build the report.
#
#   make -f latex.mk                    full report        -> report.pdf
#   make -f latex.mk only S=perception  one section/appx   -> report-perception.pdf
#   make -f latex.mk clean              remove out/ and the PDFs
#
# Every build artifact (.aux .log .toc .bbl ...) goes to out/; only the
# PDF is copied back here. The full build reruns only when a source
# changes: report.tex, preamble/, sections/, appendices/, refs.bib, figures/.

# System variables
PDFLATEX = pdflatex
BIBTEX   = bibtex

# Files and directories
MAIN    = report
OUT_DIR = out
SRCS    = $(MAIN).tex \
          $(wildcard preamble/*.tex sections/*.tex appendices/*.tex) \
          $(wildcard *.bib) $(wildcard figures/*)

FLAGS = -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory=$(OUT_DIR)

.PHONY: all only clean

all: $(MAIN).pdf

# Full build: LaTeX, BibTeX, then LaTeX twice for citations, the TOC and
# cross-references. BibTeX runs inside out/ and finds refs.bib one level up.
$(MAIN).pdf: $(SRCS) | $(OUT_DIR)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	cd $(OUT_DIR) && BIBINPUTS=..: $(BIBTEX) $(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN)
	cp $(OUT_DIR)/$(MAIN).pdf $@

# One section or appendix on its own, for fast drafting. Defines \Only
# before report.tex loads, so preamble/macros.tex skips everything else.
# References to other sections and citations show as ?? here; that's
# expected.
only: | $(OUT_DIR)
	@test -n "$(S)" || { echo "usage: make -f latex.mk only S=<name>"; exit 1; }
	@test -f sections/$(S).tex || test -f appendices/$(S).tex || \
		{ echo "no sections/$(S).tex or appendices/$(S).tex"; exit 1; }
	for i in 1 2; do \
		$(PDFLATEX) $(FLAGS) -jobname=$(MAIN)-$(S) "\def\Only{$(S)}\input{$(MAIN)}" || exit 1; \
	done
	cp $(OUT_DIR)/$(MAIN)-$(S).pdf .

$(OUT_DIR):
	mkdir -p $@

clean:
	rm -rf $(OUT_DIR) $(MAIN).pdf $(MAIN)-*.pdf
