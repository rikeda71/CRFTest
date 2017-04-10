import method
from itertools import chain
import pycrfsuite
import scipy
import sklearn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import codecs


if __name__ == '__main__':
    c = CorpusReader.CorpusReader('hironsan.txt')
    train_sents = c.iob_sents('train')
    test_sents = c.iob_sents('test')
    #print(test_sents[0])
    #print(method.sent2features(train_sents[0])[0])
    X_train = [method.sent2features(s) for s in train_sents]
    y_train = [method.sent2labels(s) for s in train_sents]
    X_test = [method.sent2features(s) for s in test_sents]
    y_test = [method.sent2labels(s) for s in test_sents]

    iner = pycrfsuite.Trainer(verbose=False)

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
