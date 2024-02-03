import os
import csv
import sys
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import utils

FEATURE_NUM = {
    'IS09_emotion': 384,
    'IS10_paraling': 1582,
    'IS11_speaker_state': 4368,
    'IS12_speaker_trait': 6125,
    'IS13_ComParE': 6373,
    'ComParE_2016': 6373
}

def get_feature_opensmile(config, filepath: str) -> list:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    single_feat_path = os.path.join(BASE_DIR, config.feature_folder, 'single_feature.csv')
    opensmile_config_path = os.path.join(config.opensmile_path, 'config', config.opensmile_config + '.conf')
    cmd = 'cd ' + config.opensmile_path + ' && ./SMILExtract -C ' + opensmile_config_path + ' -I ' + filepath + ' -O ' + single_feat_path + ' -appendarff 0'
    print("Opensmile cmd: ", cmd)
    os.system(cmd)

    reader = csv.reader(open(single_feat_path,'r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    return last_line[1: FEATURE_NUM[config.opensmile_config] + 1]

def load_feature(config, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    feature_path = os.path.join(config.feature_folder, "train.csv" if train == True else "predict.csv")
    df = pd.read_csv(feature_path)
    features = [str(i) for i in range(1, FEATURE_NUM[config.opensmile_config] + 1)]

    X = df.loc[:,features].values
    Y = df.loc[:,'label'].values

    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_OPENSMILE.m')

    if train == True:
        scaler = StandardScaler().fit(X)
        utils.mkdirs(config.checkpoint_path)
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        return x_train, x_test, y_train, y_test
    else:
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X

def get_data(config, data_path: str, train: bool) -> Union[Tuple[np.ndarray], np.ndarray]:
    utils.mkdirs(config.feature_folder)
    feature_path = os.path.join(config.feature_folder, "train.csv" if train == True else "predict.csv")
    writer = csv.writer(open(feature_path, 'w'))
    first_row = ['label']
    for i in range(1, FEATURE_NUM[config.opensmile_config] + 1):
        first_row.append(str(i))
    writer.writerow(first_row)

    writer = csv.writer(open(feature_path, 'a+'))
    print('Opensmile extracting...')

    if train == True:
        cur_dir = os.getcwd()
        sys.stderr.write('Curdir: %s\n' % cur_dir)
        os.chdir(data_path)

        for i, directory in enumerate(config.class_labels):
            sys.stderr.write("Started reading folder %s\n" % directory)
            os.chdir(directory)

            label = config.class_labels.index(directory)

            for filename in os.listdir('.'):
                if not filename.endswith('wav'):
                    continue
                filepath = os.path.join(os.getcwd(), filename)
                feature_vector = get_feature_opensmile(config, filepath)
                feature_vector.insert(0, label)
                writer.writerow(feature_vector)

            sys.stderr.write("Ended reading folder %s\n" % directory)
            os.chdir('..')
        os.chdir(cur_dir)

    else:
        feature_vector = get_feature_opensmile(config, data_path)
        feature_vector.insert(0, '-1')
        writer.writerow(feature_vector)

    print('Opensmile extract done.')
    if train == True:
        return load_feature(config, train=train)
