#coding:utf-8
# method
import method
import Reader
import report
#crf
import pycrfsuite
import sklearn

if __name__ == '__main__':
    c = Reader.CorpusReader('hironsan.txt')
    train_sents = c.iob_sents('train')
    test_sents = c.iob_sents('test')
    #print(test_sents[0])
    #print(method.sent2features(train_sents[0])[0])
    X_train = [method.sent2features(s) for s in train_sents]
    y_train = [method.sent2labels(s) for s in train_sents]
    X_test = [method.sent2features(s) for s in test_sents]
    y_test = [method.sent2labels(s) for s in test_sents]

    trainer = pycrfsuite.Trainer(verbose=False)

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

    # タグ付け
    example_sent = test_sents[0]
    print(' '.join(method.sent2tokens(example_sent)))
    print('Predicted:', ' '.join(tagger.tag(method.sent2features(example_sent))))
    print('Correct:  ', ' '.join(method.sent2labels(example_sent)))
    
    # 評価
    y_pred = [tagger.tag(xseq) for xseq in X_test]
    print(report.bio_classification_report(y_test, y_pred))
