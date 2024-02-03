import os
import numpy as np
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import models
import utils

def predict(config, audio_path: str, model) -> None:
    if config.feature_method == 'o':
        of.get_data(config, audio_path, train=False)
        test_feature = of.load_feature(config, train=False)
    elif config.feature_method == 'l':
        test_feature = lf.get_data(config, audio_path, train=False)

    result = model.predict(test_feature)
    result_prob = model.predict_proba(test_feature)
    print('Recogntion: ', config.class_labels[int(result)])
    print('Probability: ', result_prob)
    utils.radar(result_prob, config.class_labels)


if __name__ == '__main__':
    audio_path = '/Users/zou/Renovamen/Developing/Speech-Emotion-Recognition/datasets/CASIA/angry/201-angry-liuchanhg.wav'

    config = utils.parse_opt()
    model = models.load(config)
    predict(config, audio_path, model)
