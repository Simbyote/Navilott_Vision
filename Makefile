# vision_stack/Makefile
#
# Three stages, usable standalone or chained:
#
#   1. sweep-singles / sweep-combo / sweep-baseline   calls sweep_fix.sh / sweep_fix_combo.sh / sweep_baseline.sh
#   2. analyze                                        calls combine_runs.py then plot_runs.py
#   3. test-singles / test-combo / test-baseline      stage 1 target then stage 2 target
#
# `make help` prints the target list and examples below.

LOG_DIR      := vision_stack/logs
COMBINED_DIR := $(LOG_DIR)/combined
PLOTS_DIR    := $(LOG_DIR)/plots
SCRIPTS_DIR  := vision_stack/scripts

.DEFAULT_GOAL := help
.PHONY: help sweep-singles sweep-combo sweep-baseline analyze test-singles test-combo test-baseline

help:
	@echo "Targets:"
	@echo "  sweep-singles TRACK=<track> GEN=<gen> PARAMTAG=<tag>"
	@echo "      sweep_fix.sh: all 5 single-fix bench tests -> $(LOG_DIR)/<TRACK>Logs/{I,T,O,D,A}/<gen>/"
	@echo ""
	@echo "  sweep-combo TRACK=<track> GEN=<gen> PARAMTAG=<tag> FIXES=<fix1,fix2,...>"
	@echo "      sweep_fix_combo.sh: one bench test with several fixes on -> $(LOG_DIR)/<TRACK>Logs/<LETTERS>/<gen>/"
	@echo ""
	@echo "  sweep-baseline TRACK=<track> GEN=<gen> PARAMTAG=<tag>"
	@echo "      sweep_baseline.sh: one bench test with NO fixes on -> $(LOG_DIR)/<TRACK>Logs/NONE/<gen>/"
	@echo ""
	@echo "  analyze NAME=<name> INPUTS=\"<paths relative to $(LOG_DIR)>\""
	@echo "      combine_runs.py then plot_runs.py -> $(COMBINED_DIR)/<name>_* and $(PLOTS_DIR)/<name>/"
	@echo ""
	@echo "  test-singles TRACK=<track> GEN=<gen> PARAMTAG=<tag> [NAME=<name>]"
	@echo "      sweep-singles, then analyze over all 5 single-fix dirs under GEN. NAME defaults to PARAMTAG."
	@echo ""
	@echo "  test-combo TRACK=<track> GEN=<gen> PARAMTAG=<tag> FIXES=<fix1,fix2,...> [NAME=<name>]"
	@echo "      sweep-combo, then analyze over the resulting combo/GEN dir. NAME defaults to PARAMTAG."
	@echo ""
	@echo "  test-baseline TRACK=<track> GEN=<gen> PARAMTAG=<tag> [NAME=<name>]"
	@echo "      sweep-baseline, then analyze over the NONE/GEN dir. NAME defaults to PARAMTAG."
	@echo ""
	@echo "GEN identifies the config.py baseline (non-swept dataclass defaults) a batch"
	@echo "of runs was taken against. Bump it by hand when those defaults change; log"
	@echo "the change in $(LOG_DIR)/CALIBRATION_LOG.md. See FILENAME_LEGEND.md for PARAMTAG."
	@echo ""
	@echo "Examples:"
	@echo "  make test-baseline TRACK=T3 GEN=gen0 PARAMTAG=baseline"
	@echo "  make test-singles TRACK=T3 GEN=gen0 PARAMTAG=sm15_tf55"
	@echo "  make test-combo TRACK=T3 GEN=gen0 PARAMTAG=oa65_dh12 FIXES=orientation_filt,dashed_dilate"
	@echo "  make analyze NAME=od_family INPUTS=\"T3Logs/OD/gen0 T3Logs/OA/gen0 T3Logs/DA/gen0\""

# ---------------------------------------------------------------------------
# Stage 1: sweeps — generate the CSVs
# ---------------------------------------------------------------------------
sweep-singles:
ifndef TRACK
	$(error TRACK is required, e.g. make sweep-singles TRACK=T3 GEN=gen0 PARAMTAG=sm15_tf55)
endif
ifndef GEN
	$(error GEN is required, e.g. make sweep-singles TRACK=T3 GEN=gen0 PARAMTAG=sm15_tf55)
endif
ifndef PARAMTAG
	$(error PARAMTAG is required, e.g. make sweep-singles TRACK=T3 GEN=gen0 PARAMTAG=sm15_tf55)
endif
	./$(SCRIPTS_DIR)/sweep_fix.sh $(TRACK) $(GEN) $(PARAMTAG)

sweep-combo:
ifndef TRACK
	$(error TRACK is required, e.g. make sweep-combo TRACK=T3 GEN=gen0 PARAMTAG=oa65_dh12 FIXES=orientation_filt,dashed_dilate)
endif
ifndef GEN
	$(error GEN is required, e.g. make sweep-combo TRACK=T3 GEN=gen0 PARAMTAG=oa65_dh12 FIXES=orientation_filt,dashed_dilate)
endif
ifndef PARAMTAG
	$(error PARAMTAG is required, e.g. make sweep-combo TRACK=T3 GEN=gen0 PARAMTAG=oa65_dh12 FIXES=orientation_filt,dashed_dilate)
endif
ifndef FIXES
	$(error FIXES is required, e.g. make sweep-combo TRACK=T3 GEN=gen0 PARAMTAG=oa65_dh12 FIXES=orientation_filt,dashed_dilate)
endif
	./$(SCRIPTS_DIR)/sweep_fix_combo.sh $(TRACK) $(GEN) $(PARAMTAG) $(FIXES)

sweep-baseline:
ifndef TRACK
	$(error TRACK is required, e.g. make sweep-baseline TRACK=T3 GEN=gen0 PARAMTAG=baseline)
endif
ifndef GEN
	$(error GEN is required, e.g. make sweep-baseline TRACK=T3 GEN=gen0 PARAMTAG=baseline)
endif
ifndef PARAMTAG
	$(error PARAMTAG is required, e.g. make sweep-baseline TRACK=T3 GEN=gen0 PARAMTAG=baseline)
endif
	./$(SCRIPTS_DIR)/sweep_baseline.sh $(TRACK) $(GEN) $(PARAMTAG)

# ---------------------------------------------------------------------------
# Stage 2: analyze — combine + plot a chosen set of runs
# ---------------------------------------------------------------------------
analyze:
ifndef NAME
	$(error NAME is required, e.g. make analyze NAME=baseline_singles INPUTS="T3Logs/NONE T3Logs/I")
endif
ifndef INPUTS
	$(error INPUTS is required, e.g. make analyze NAME=baseline_singles INPUTS="T3Logs/NONE T3Logs/I")
endif
	python3 $(SCRIPTS_DIR)/combine_runs.py \
		--out $(COMBINED_DIR)/$(NAME) \
		$(addprefix $(LOG_DIR)/,$(INPUTS))
	@if [ -f "$(COMBINED_DIR)/$(NAME)_contours.csv" ]; then \
		python3 $(SCRIPTS_DIR)/plot_runs.py \
			--frames $(COMBINED_DIR)/$(NAME)_frames.csv \
			--contours $(COMBINED_DIR)/$(NAME)_contours.csv \
			--out $(PLOTS_DIR)/$(NAME); \
	else \
		echo "NOTE: no run in this batch had DEBUG_CONTOURS data — skipping contours plot"; \
		python3 $(SCRIPTS_DIR)/plot_runs.py \
			--frames $(COMBINED_DIR)/$(NAME)_frames.csv \
			--out $(PLOTS_DIR)/$(NAME); \
	fi
	@echo "Done: $(COMBINED_DIR)/$(NAME)_* -> $(PLOTS_DIR)/$(NAME)/"

# ---------------------------------------------------------------------------
# Stage 3: sweep + analyze in one call
# ---------------------------------------------------------------------------

# sweep_fix.sh writes to the I, T, O, D, A directories, each nested under
# GEN. INPUTS is built from TRACK and GEN using those five letters.
test-singles: sweep-singles
	$(MAKE) analyze \
		NAME=$(if $(NAME),$(NAME),$(PARAMTAG)) \
		INPUTS="$(TRACK)Logs/I/$(GEN) $(TRACK)Logs/T/$(GEN) $(TRACK)Logs/O/$(GEN) $(TRACK)Logs/D/$(GEN) $(TRACK)Logs/A/$(GEN)"

# sweep_fix_combo.sh names its output directory using the canonical-order
# letters of FIXES (roi_inset=I, trapezoid_mask=T, orientation_filt=O,
# dashed_dilate=D, anchor_halves=A). The same mapping is computed here to
# build INPUTS.
test-combo: sweep-combo
	@LETTERS=$$(python3 -c "canon=['roi_inset','trapezoid_mask','orientation_filt','dashed_dilate','anchor_halves']; letters=['I','T','O','D','A']; requested=set('$(FIXES)'.split(',')); print(''.join(l for f,l in zip(canon,letters) if f in requested))"); \
	$(MAKE) analyze \
		NAME=$(if $(NAME),$(NAME),$(PARAMTAG)) \
		INPUTS="$(TRACK)Logs/$$LETTERS/$(GEN)"

# sweep_baseline.sh always writes to the NONE directory (no --fix flag
# passed to run_pipeline.py). INPUTS is built from TRACK and GEN.
test-baseline: sweep-baseline
	$(MAKE) analyze \
		NAME=$(if $(NAME),$(NAME),$(PARAMTAG)) \
		INPUTS="$(TRACK)Logs/NONE/$(GEN)"