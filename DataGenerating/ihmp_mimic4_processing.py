from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import pickle
import numpy as np
from DataGenerating import load_data_utils
from DataGenerating.preprocessing import Discretizer
from DataGenerating.readers import InHospitalMortalityReader


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default='/home/xxx/mimic4processed/in-hospital-mortality/')
args = parser.parse_args()
print(args)
# Build readers, discretizers, normalizers
train_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'),
                                         period_length=48.0)

val_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'),
                                       period_length=48.0)

test_reader = InHospitalMortalityReader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'),
                                        period_length=48.0)

discretizer = Discretizer(timestep=float(1.0),
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

print('Train:')
train_raw, normalizer = load_data_utils.ihmp_load_data(dataloader=train_reader, discretizer=discretizer, normalizer=None)
print('\nVal:')
val_raw = load_data_utils.ihmp_load_data(dataloader=val_reader, discretizer=discretizer, normalizer=normalizer)
print('\nTest:')
test_raw = load_data_utils.ihmp_load_data(dataloader=test_reader, discretizer=discretizer, normalizer=normalizer)

train_x = np.array(train_raw[0], dtype=float)
train_mask = np.ones((train_x.shape[0], train_x.shape[1]), dtype=float)
train_cur_mask = np.zeros((train_x.shape[0], train_x.shape[1]), dtype=float)
train_cur_mask[:, -1] = 1.0
print(train_mask[:5])
print(train_cur_mask[:5])
train_dt = np.ones((train_x.shape[0], train_x.shape[1]), dtype=float)
train_y = np.expand_dims(np.array(train_raw[1], dtype=float), axis=-1)
train_dict = {'X': train_x, 'MASK': train_mask, 'CUR_MASK': train_cur_mask, 'INTERVAL': train_dt, 'Y': train_y, 'NAME': train_raw[2]}


val_x = np.array(val_raw[0], dtype=float)
val_mask = np.ones((val_x.shape[0], val_x.shape[1]), dtype=float)
val_cur_mask = np.zeros((val_x.shape[0], val_x.shape[1]), dtype=float)
val_cur_mask[:, -1] = 1.0
val_dt = np.ones((val_x.shape[0], val_x.shape[1]), dtype=float)
val_y = np.expand_dims(np.array(val_raw[1], dtype=float), axis=-1)
val_dict = {'X': val_x, 'MASK': val_mask, 'CUR_MASK': val_cur_mask, 'INTERVAL': val_dt, 'Y': val_y, 'NAME': val_raw[2]}


test_x = np.array(test_raw[0], dtype=float)
test_mask = np.ones((test_x.shape[0], test_x.shape[1]), dtype=float)
test_cur_mask = np.zeros((test_x.shape[0], test_x.shape[1]), dtype=float)
test_cur_mask[:, -1] = 1.0
test_dt = np.ones((test_x.shape[0], test_x.shape[1]), dtype=float)
test_y = np.expand_dims(np.array(test_raw[1], dtype=float), axis=-1)
test_dict = {'X': test_x, 'MASK': test_mask, 'CUR_MASK': test_cur_mask, 'INTERVAL': test_dt, 'Y': test_y, 'NAME': test_raw[2]}

print(train_x.shape, train_mask.shape, train_dt.shape, train_y.shape, '\n',
      val_x.shape, val_mask.shape, val_dt.shape, val_y.shape, '\n',
      test_x.shape, test_mask.shape, test_dt.shape, test_y.shape)

#mimic4
#  (18900, 48, 76) (18900, 48) (18900, 48) (18900, 1)
#  (3435, 48, 76) (3435, 48) (3435, 48) (3435, 1)
#  (3891, 48, 76) (3891, 48) (3891, 48) (3891, 1)

pickle_file = open('/home/xxx/ehr_data/ihmp/ihmp_mimic4_train_data.pkl', 'wb')
pickle.dump(train_dict, pickle_file)
pickle_file.close()

pickle_file = open('/home/xxx/ehr_data/ihmp/ihmp_mimic4_val_data.pkl', 'wb')
pickle.dump(val_dict, pickle_file)
pickle_file.close()

pickle_file = open('/home/xxx/ehr_data/ihmp/ihmp_mimic4_test_data.pkl', 'wb')
pickle.dump(test_dict, pickle_file)
pickle_file.close()