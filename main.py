#coding:utf-8
# method
import method
import Reader
import report
#crf
import pycrfsuite
import sklearn
# closs-validation
from sklearn.cross_validation import KFold

if __name__ == '__main__':
    c = Reader.CorpusReader('hironsan.txt')
    all_sents = c.iob_sents('all')
    
    k_fold = KFold(n=len(all_sents), n_folds = 10, shuffle=True)
    for train, test in k_fold:
        trainer = pycrfsuite.Trainer(verbose=False)
        X_train = [method.sent2features(all_sents[s]) for s in train]
        y_train = [method.sent2labels(all_sents[s]) for s in train]
        X_test = [method.sent2features(all_sents[s]) for s in test]
        y_test = [method.sent2labels(all_sents[s]) for s in test]
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train('model.crfsuite')

        # テストデータの予測
        tagger = pycrfsuite.Tagger()
        tagger.open('model.crfsuite')

        # 評価
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        print(report.bio_classification_report(y_test, y_pred))
