# packages
APT_PKGS := python-pip python-dev
BREW_PKGS := --python
PIP_PKGS := numpy scipy pandas scikit-learn

SED := gsed

# directories
DIR_DATA := data
DIR_BUILD := build
DIR_SRC := src
DIR_BIN := $(DIR_BUILD)/bin
DIR_BLEND := $(DIR_BUILD)/blend
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

# data files after clean-up, transformation, join etc.
DATA_OBJ_UNIQ := $(DIR_DATA)/object_uniq.csv
DATA_TRN_LOG_ID := $(DIR_DATA)/log_train_id.csv
DATA_TST_LOG_ID := $(DIR_DATA)/log_test_id.csv
DATA_TRN_LOG_V2 := $(DIR_DATA)/log_train_source_event.csv
DATA_TST_LOG_V2 := $(DIR_DATA)/log_test_source_event.csv

DIRS := $(DIR_DATA) $(DIR_BUILD) $(DIR_FEATURE) $(DIR_METRIC) $(DIR_MODEL) \
        $(DIR_VAL) $(DIR_TST) $(DIR_BIN) $(DIR_BLEND)

# data files for training and predict
SUBMISSION_SAMPLE := $(DIR_DATA)/sampleSubmission.csv

ID_TST := $(DIR_DATA)/id.tst.txt

Y_TRN := $(DIR_DATA)/y.trn.yht
Y_TST := $(DIR_DATA)/y.tst.yht

CID_TRN := $(DIR_DATA)/cid.trn.txt
CID_TST := $(DIR_DATA)/cid.tst.txt

$(DATA_TST_LOG_ID): $(DATA_TST_LOG) $(DATA_TST_ENR)
	python src/join_log_enrollment.py --log-file $< \
                                      --enrollment-file $(lastword $^) \
                                      --out-file $@

$(DATA_TRN_LOG_ID): $(DATA_TRN_LOG) $(DATA_TRN_ENR)
	python src/join_log_enrollment.py --log-file $< \
                                      --enrollment-file $(lastword $^) \
                                      --out-file $@

$(DIRS):
	mkdir -p $@

$(ID_TST): $(SUBMISSION_SAMPLE)
	cut -d, -f1 $< > $@

$(Y_TRN): $(LABEL_TRN)
	cut -d, -f2 $< > $@

$(Y_TST): $(SUBMISSION_SAMPLE)
	cut -d, -f2 $< > $@

$(DATA_OBJ_UNIQ): $(DATA_OBJ)
	tail -n +2 $< | sort | uniq > $@

$(DATA_TRN_LOG_V2): $(DATA_TRN_LOG_ID)
	$(SED) 's/source,/source_/g' $< > $@
	$(SED) -i 's/browser,/browser_/g' $@
	$(SED) -i 's/server,/server_/g' $@

$(DATA_TST_LOG_V2): $(DATA_TST_LOG_ID)
	$(SED) 's/source,/source_/g' $< > $@
	$(SED) -i 's/browser,/browser_/g' $@
	$(SED) -i 's/server,/server_/g' $@

# cleanup
clean::
	find . -name '*.pyc' -delete

clobber: clean
	-rm -rf $(DIR_DATA) $(DIR_BUILD)

.PHONY: clean clobber mac.setup ubuntu.setup apt.setup pip.setup
