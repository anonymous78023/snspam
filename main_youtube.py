"""
This script uses EGGS to model a YouTube spam dataset.
"""
import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from EGGS.eggs import EGGS
from sklearn.linear_model import LogisticRegression
from snspam_data.youtube.features.relational import pseudo_relational as pr
from snspam_data.youtube.features.relational import pgm

from collections import defaultdict
from EGGS import print_utils
from scipy.sparse import load_npz


def main(featureset='full', setting='inductive+transductive', stacks=0, joint_model='mrf'):

    assert featureset in ['limited', 'full']
    assert setting in ['transductive', 'inductive+transductive']
    assert stacks >= 0
    assert joint_model in [None, 'mrf', 'psl']

    # setup score containers
    scores = defaultdict(list)
    y_hats = defaultdict(list)
    y_true = list()

    # test models using cross-validation
    for fold in range(10):

        print('\nfold %d:' % fold)
        data_dir = 'snspam_data/youtube/processed/folds/'
        features_dir = 'snspam_data/youtube/features/independent/%s/%s/' % (featureset, setting)

        # read in feature data
        print('reading in data...')
        X_train = load_npz('%strain_data_%d.npz' % (features_dir, fold)).tocsr()
        X_val = load_npz('%sval_data_%d.npz' % (features_dir, fold)).tocsr()
        X_test = load_npz('%stest_data_%d.npz' % (features_dir, fold)).tocsr()

        # read in label data
        train_df = pd.read_csv('%strain_%d.csv' % (data_dir, fold), usecols=['com_id', 'label'])
        val_df = pd.read_csv('%sval_%d.csv' % (data_dir, fold), usecols=['com_id', 'label'])
        test_df = pd.read_csv('%stest_%d.csv' % (data_dir, fold), usecols=['com_id', 'label'])

        # extract label data
        y_train = train_df['label'].to_numpy()
        y_val = val_df['label'].to_numpy()
        y_test = test_df['label'].to_numpy()

        # extract identifier data
        target_col_train = train_df['com_id'].to_numpy()
        target_col_val = val_df['com_id'].to_numpy()
        target_col_test = test_df['com_id'].to_numpy()

        y_true.append(y_test)

        # setup hyperparameters, metrics, and models
        lr = LogisticRegression(solver='liblinear')
        metrics = [('roc_auc', roc_auc_score), ('aupr', average_precision_score)]
        models = [
            ('lr', EGGS(estimator=lr), None),
            ('eggs', EGGS(estimator=lr,
                          sgl_method='holdout', stacks=stacks, sgl_func=pr.pseudo_relational_features,
                          joint_model=joint_model, pgm_func=pgm.create_files,
                          relations=['text_id', 'user_id'], verbose=1), None)
            ]

        # train and predict for each model
        for name, model, param_grid in models:
            print('[%s] fitting and predicting...' % name)
            model = model.fit(X_train, y_train, target_col_train, X_val, y_val, target_col_val, fold=fold)
            y_hat = model.predict_proba(X_test, target_col_test, fold=fold)[:, 1]
            y_hats[name].append(y_hat)

    # combine predictions from all folds and generate scores
    print('combining predictions...')
    y_true = np.hstack(y_true)

    for name, model, _ in models:
        y_hat = np.hstack(y_hats[name])

        for metric, scorer in metrics:
            scores[name + '|' + metric].append(scorer(y_true, y_hat))

    print_utils.print_scores(scores, models, metrics)

if __name__ == '__main__':

    # read in commandline args
    parser = argparse.ArgumentParser(description='EGGS: Extended Group-based Graphical models for Spam', prog='run')
    parser.add_argument('--featureset', default='full', help='independent features, default: %(default)s')
    parser.add_argument('--setting', default='inductive+transductive', help='network setting, default: %(default)s')
    parser.add_argument('--stacks', default=0, type=int, help='number of SGL stacks, default: %(default)s')
    parser.add_argument('--joint', default=None, help='joint inference model (mrf or psl), default: %(default)s')
    args = parser.parse_args()

    # run experiment
    main(featureset=args.featureset, setting=args.setting, stacks=args.stacks, joint_model=args.joint)
