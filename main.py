# coding:utf-8
# method
import method
import Reader
import report
# crf
import pycrfsuite
# import sklearn_crfsuite
# import sklearn
# closs-validation
from sklearn.cross_validation import KFold

if __name__ == '__main__':
    # c = Reader.CorpusReader('hironsan.txt')
    c = Reader.CorpusReader('hironsan.txt')
    all_sents = c.iob_sents('all')

    k_fold = KFold(n=len(all_sents), n_folds=10, shuffle=True)
    # K_fold = KFold(n=len(all_sents), n_folds = 10, shuffle=False)
    for train, test in k_fold:
        trainer = pycrfsuite.Trainer(verbose=False)
        # trainer = sklearn_crfsuite.Trainer(verbose=False)
        X_train = [method.sent2features(all_sents[s]) for s in train]
        y_train = [method.sent2labels(all_sents[s]) for s in train]
        # X_test = [method.sent2features(all_sents[s]) for s in test]
        y_test = [method.sent2labels(all_sents[s]) for s in test]
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        # モデルの学習
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        trainer.train('model.crfsuite')

        # 学習データの呼び出し
        tagger = pycrfsuite.Tagger()
        tagger.open('model.crfsuite')

        # タグの予測
        # メソッドの使い方の実験
        # 素性を与えるときに、タグの情報が予想したものでなく、
        # 正解を与えているが、それでも大丈夫なのか
        y_pred = []
        for i in range(len(test)):
            array = []
            true_array = []
            for j in range(len(all_sents[test[i]])):
                if j == 0:
                    X_test = method.sent2features(all_sents[test[i]])
                else:
                    X_test = method.sent2prediction(all_sents[test[i]], j, array)
                array = tagger.tag(X_test)
                true_array.append(array[j])
            y_pred.append(true_array)

        # 評価
        # y_pred = [tagger.tag(xseq) for xseq in X_test]
        print("オリジナル")
        print(report.bio_classification_report(y_test, y_pred))

        # サンプル通り
        print("ネットでの使い方")
        y_pred = [tagger.tag(xseq) for xseq in [method.sent2features(all_sents[s]) for s in test]]
        print(report.bio_classification_report(y_test, y_pred))
