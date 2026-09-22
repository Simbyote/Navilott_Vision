# =============================================================================
# Navilott_Vision — Pi Zero 2W Bootrapper
# =============================================================================
#   make setup          # everything except the final reboot
#   make reboot         # reboot once you're happy
#   make all            # setup + reboot, no pause
#
# Targets are idempotent
# Run `make help` to see all targets.
# =============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

# --- Custom Paths --------------------------------------------------
VENV_DIR      ?= $(HOME)/.venv/navilott
PIGPIO_DIR    ?= $(HOME)/pigpio
PIGPIO_TAG    ?= v79
PROJECT_DIR   ?= $(HOME)/Navilott_Vision

# Marker files
STAMP_DIR     := .make-stamps
APT_STAMP     := $(STAMP_DIR)/apt-base
CAM_STAMP     := $(STAMP_DIR)/apt-camera
I2C_STAMP     := $(STAMP_DIR)/i2c-enabled
PIGPIO_STAMP  := $(STAMP_DIR)/pigpio-installed

.PHONY: all setup help update base-deps camera-deps check i2c pigpio venv install reboot clean distclean

# --- Meta ------------------------------------------------------------------

help:
	@echo "Targets:"
	@echo "  make setup         - run everything except reboot"
	@echo "  make all           - setup + reboot immediately"
	@echo "  make update        - apt update && full-upgrade"
	@echo "  make base-deps     - base build/dev tools"
	@echo "  make camera-deps   - libcamera/gstreamer/opencv stack"
	@echo "  make check         - verify camera stack is actually working"
	@echo "  make i2c           - enable I2C via raspi-config"
	@echo "  make pigpio        - clone, build, install pigpiod as a service"
	@echo "  make venv          - create venv + pip install -e . for this repo"
	@echo "  make reboot        - reboot the Pi"
	@echo "  make clean         - remove build artifacts (keeps venv)"
	@echo "  make distclean     - clean + remove venv + stamps (full reset)"

setup: update base-deps camera-deps i2c pigpio venv check
	@echo ""
	@echo "==> Setup complete. Review 'make check' output above,"
	@echo "    then run 'make reboot' when ready."

all: setup reboot

$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)

# --- Base System -------------------------------------------------------

update:
	sudo apt update && sudo apt full-upgrade -y

base-deps: $(STAMP_DIR)
	@if [ -f $(APT_STAMP) ]; then \
		echo "==> base-deps already installed, skipping (rm $(APT_STAMP) to force)"; \
	else \
		sudo apt install -y git tmux build-essential i2c-tools \
			python3-pip python3-numpy python3-opencv; \
		touch $(APT_STAMP); \
	fi

# --- Camera Stack --------------------------------------------------------

camera-deps: $(STAMP_DIR)
	@if [ -f $(CAM_STAMP) ]; then \
		echo "==> camera-deps already installed, skipping (rm $(CAM_STAMP) to force)"; \
	else \
		sudo apt install -y libcamera-apps \
			gstreamer1.0-tools gstreamer1.0-libcamera \
			gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
			gstreamer1.0-plugins-bad \
			python3-gi python3-gst-1.0; \
		touch $(CAM_STAMP); \
	fi

# Check Commands
check:
	@echo "--- rpicam-hello ---"
	-rpicam-hello --list-cameras
	@echo "--- gstreamer libcamerasrc ---"
	-gst-inspect-1.0 libcamerasrc | head -5
	@echo "--- opencv gstreamer support ---"
	-python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer

# --- I2C ---------------------------------------------------------------

i2c: $(STAMP_DIR)
	@if [ -f $(I2C_STAMP) ]; then \
		echo "==> I2C already enabled, skipping"; \
	else \
		sudo raspi-config nonint do_i2c 0; \
		touch $(I2C_STAMP); \
	fi

# --- pigpio --------------------------------------------------------------

pigpio: $(STAMP_DIR)
	@if [ -f $(PIGPIO_STAMP) ]; then \
		echo "==> pigpio already installed, skipping (rm $(PIGPIO_STAMP) to force)"; \
	else \
		if [ -d $(PIGPIO_DIR) ]; then \
			echo "==> $(PIGPIO_DIR) exists, reusing checkout"; \
		else \
			git clone https://github.com/joan2937/pigpio.git $(PIGPIO_DIR); \
		fi; \
		cd $(PIGPIO_DIR) && git checkout $(PIGPIO_TAG) && $(MAKE); \
		sudo $(MAKE) -C $(PIGPIO_DIR) install; \
		pip3 install pigpio --break-system-packages; \
		sudo ldconfig; \
		sudo cp $(PIGPIO_DIR)/util/pigpiod.service /etc/systemd/system/; \
		sudo systemctl daemon-reload; \
		sudo systemctl enable --now pigpiod; \
		touch $(PIGPIO_STAMP); \
	fi

# Check Commands
	@echo "--- pigpiod status ---"
	-which pigpiod
	-ldconfig -p | grep pigpio
	-systemctl status pigpiod --no-pager
	-pigs t

# --- Python Environment Install --------------------------------

venv:
	@if [ -d $(VENV_DIR) ]; then \
		echo "==> venv already exists at $(VENV_DIR), skipping creation"; \
	else \
		python3 -m venv $(VENV_DIR) --system-site-packages; \
	fi
	source $(VENV_DIR)/bin/activate && \
		cd $(PROJECT_DIR) && \
		pip install -e .
	@echo ""
	@echo "==> Remember: 'source $(VENV_DIR)/bin/activate' in new shells."

# --- Cleaning ----------------------------------------------------------

reboot:
	sudo reboot

clean:
	rm -rf $(PIGPIO_DIR)

distclean: clean
	rm -rf $(VENV_DIR) $(STAMP_DIR)