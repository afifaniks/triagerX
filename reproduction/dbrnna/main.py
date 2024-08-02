# Reference Repository: https://github.com/hacetin/deep-triage

import tensorflow
import tensorflow as tf

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
print('############## Allowing Growth ###########')
session = tf.Session(config=config)


from dbrnna import run_dbrnna_chronological_cv
from preprocess import preprocess_all_datasets


preprocess_all_datasets()

gc_result_dict = run_dbrnna_chronological_cv(
    dataset_name="google_chromium",
    min_train_samples_per_class=20,
    num_cv=10,
    rnn_type="lstm",
)
print("gc_result_dict:", gc_result_dict)

mc_result_dict = run_dbrnna_chronological_cv(
    dataset_name="mozilla_core", min_train_samples_per_class=20, num_cv=10
)
print("mc_result_dict:", mc_result_dict)

mf_result_dict = run_dbrnna_chronological_cv(
    dataset_name="mozilla_firefox", min_train_samples_per_class=0, num_cv=10
)
print("mf_result_dict:", mf_result_dict)

openj9_result_dict = run_dbrnna_chronological_cv(
    dataset_name="openj9", min_train_samples_per_class=20, num_cv=10
)
print("openj9_result_dict:", openj9_result_dict)

ts_result_dict = run_dbrnna_chronological_cv(
    dataset_name="typescript", min_train_samples_per_class=20, num_cv=10
)
print("ts_result_dict:", ts_result_dict)
