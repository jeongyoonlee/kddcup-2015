include Makefile.feature.feature9

N = 2
LRATE = 0.1
ALGO_NAME := vw_$(N)_$(LRATE)
MODEL_NAME := $(ALGO_NAME)_$(FEATURE_NAME)

METRIC_VAL := $(DIR_METRIC)/$(MODEL_NAME).val.txt

MODEL_TRN := $(DIR_MODEL)/$(MODEL_NAME).trn.mdl

PREDICT_VAL := $(DIR_VAL)/$(MODEL_NAME).val.yht
PREDICT_TST := $(DIR_TST)/$(MODEL_NAME).tst.yht

SUBMISSION_TST := $(DIR_TST)/$(MODEL_NAME).sub.csv
SUBMISSION_TST_GZ := $(DIR_TST)/$(MODEL_NAME).sub.csv.gz

all: validation submission
validation: $(METRIC_VAL)
submission: $(SUBMISSION_TST)
retrain: clean_$(ALGO_NAME) submission

$(PREDICT_TST) $(PREDICT_VAL) $(MODEL_TRN): $(FEATURE_TRN_VW) \
                                            $(FEATURE_TST_VW) \
                                            $(FEATURE_TRN) \
                                   | $(DIR_VAL) $(DIR_TST) $(DIR_MODEL)
	./src/train_predict_vw.py --train-file $< \
                              --test-file $(word 2, $^) \
                              --train-svm-file $(lastword $^) \
                              --predict-valid-file $(PREDICT_VAL) \
                              --predict-test-file $(PREDICT_TST) \
                              --model-file $(MODEL_TRN) \
                              --lrate $(LRATE) \
                              --n-iter $(N)

$(SUBMISSION_TST_GZ): $(SUBMISSION_TST)
	gzip $<

$(SUBMISSION_TST): $(PREDICT_TST) $(ID_TST) | $(DIR_TST)
	paste -d, $(lastword $^) $< > $@

$(METRIC_VAL): $(PREDICT_VAL) $(LABEL_TRN) | $(DIR_METRIC)
	./src/evaluate.py -t $(lastword $^) -p $< > $@
	cat $@


clean:: clean_$(ALGO_NAME)

clean_$(ALGO_NAME):
	-rm $(METRIC_VAL) $(PREDICT_VAL) $(PREDICT_TST) $(SUBMISSION_TST)
	find . -name '*.pyc' -delete

.DEFAULT_GOAL := all
