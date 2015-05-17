# packages
APT_PKGS := python-pip python-dev
BREW_PKGS := --python
PIP_PKGS := numpy scipy pandas scikit-learn

# directories
DIR_DATA := data
DIR_BUILD := build
DIR_SRC := src
DIR_BIN := $(DIR_BUILD)/bin
DIR_FEATURE := $(DIR_BUILD)/feature
DIR_METRIC := $(DIR_BUILD)/metric
DIR_MODEL := $(DIR_BUILD)/model

# directories for the cross validation and ensembling
DIR_VAL := $(DIR_BUILD)/val
DIR_TST := $(DIR_BUILD)/tst

DATA_TRN_LOG := $(DIR_DATA)/log_train.csv
DATA_TST_LOG := $(DIR_DATA)/log_test.csv
DATA_TRN_ENR := $(DIR_DATA)/enrollment_train.csv
DATA_TST_ENR := $(DIR_DATA)/enrollment_test.csv
DATA_OBJ := $(DIR_DATA)/object.csv
LABEL_TRN := $(DIR_DATA)/truth_train.csv

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_BIN)

# data files for training and predict
SUBMISSION_SAMPLE := $(DIR_DATA)/sampleSubmission.csv

ID_TST := $(DIR_DATA)/id.tst.txt

$(DIRS):
	mkdir -p $@

$(ID_TST) : $(SUBMISSION_SAMPLE)
	cut -d, -f1 $< > $@

# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
