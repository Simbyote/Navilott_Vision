# vision_stack/Makefile
#
# analyze: combine a set of run folders/zips into one dataset, then plot it.
# Usage:
#   make analyze NAME=baseline_singles INPUTS="T3Logs/NONE T3Logs/I T3Logs/T T3Logs/O T3Logs/D T3Logs/A"
#   make analyze NAME=od_family INPUTS="T3Logs/OD T3Logs/OA T3Logs/DA"
#
# NAME is shared between logs/combined/<NAME>_* and logs/plots/<NAME>/ so the
# two directories always mirror each other — same convention as the sweep
# scripts' <paramtag>__<timestamp> naming, just one level up.
#
# INPUTS are paths relative to LOG_DIR (so you don't retype the log root each
# time); mix folders and .zip files freely, same as combine_runs.py itself.

LOG_DIR      := vision_stack/logs
COMBINED_DIR := $(LOG_DIR)/combined
PLOTS_DIR    := $(LOG_DIR)/plots
SCRIPTS_DIR  := vision_stack/scripts

.PHONY: analyze
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