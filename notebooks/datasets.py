import tensorflow as tf
from functools import partial


autotune_ = tf.data.AUTOTUNE

higgs_features_names = [
   'lepton_pT',
   'lepton_eta',
   'lepton_phi',
   'missing_energy_magnitude',
   'missing_energy_phi',
   'jet1pt',
   'jet1eta',
   'jet1phi',
   'jet1b-tag',
   'jet2pt',
   'jet2eta',
   'jet2phi',
   'jet2b-tag',
   'jet3pt',
   'jet3eta',
   'jet3phi',
   'jet3b-tag',
   'jet4pt',
   'jet4eta',
   'jet4phi',
   'jet4b-tag',
   'm_jj',
   'm_jjj',
   'm_lv',
   'm_jlv',
   'm_bb',
   'm_wbb',
   'm_wwbb',
]

meson_features_names = [
     'mc_E[0]',
     'mc_E[1]',
     'mc_E[2]',
     'mc_E[3]',
     'mc_Px[0]',
     'mc_Px[1]',
     'mc_Px[2]',
     'mc_Px[3]',
     'mc_Py[0]',
     'mc_Py[1]',
     'mc_Py[2]',
     'mc_Py[3]',
     'mc_Pz[0]',
     'mc_Pz[1]',
     'mc_Pz[2]',
     'mc_Pz[3]',
     'mc_wD',
]

features_names_ = {
    'higgs': higgs_features_names,
    'meson': meson_features_names,
}


def _parse_function(example_proto, with_label=True, features_names=higgs_features_names):
    # Create a description of the features.
    feature_description = {
        feature_name: tf.io.FixedLenFeature([], tf.float32, default_value=0)
        for feature_name in features_names
    }
    if with_label:
        feature_description['label'] = tf.io.FixedLenFeature([], tf.float32, default_value=0)

    # Parse the input `tf.train.Example` proto using the dictionary above.
    example = tf.io.parse_single_example(example_proto, feature_description)
    X = [example[feature_name] for feature_name in features_names]
    if with_label:
        y = example['label']
        return X, y
    return X,

def get_dataset(filenames_template="/mnt/storage-large/dataset/higgs/higgs_tfrecords/test/*.tfrecord", with_label=True, batch_size=32, seed=None, dataset='higgs'):
    filenames = tf.io.gfile.glob(filenames_template)
    raw_dataset = tf.data.TFRecordDataset(filenames)
    features_names = features_names_[dataset]
    parser = partial(_parse_function, with_label=with_label, features_names=features_names)
    dataset = raw_dataset.map(parser, num_parallel_calls=autotune_)
    if seed is not None:
        dataset = dataset.shuffle(seed)
    dataset = dataset.prefetch(buffer_size=autotune_)
    dataset = dataset.batch(batch_size)
    return dataset
