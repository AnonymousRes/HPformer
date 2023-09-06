from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import pickle
import numpy as np
from DataGenerating import common_utils
from DataGenerating import load_data_utils
from DataGenerating.preprocessing import Discretizer
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data of decompensation task',
                    default='/home/xxx/mimic3processed/decompensation/')
args = parser.parse_args()
# weeks = 24
print(args)
timestep_ = 1  # hours
print('timestep:', timestep_)
# print('weeks', weeks)
# 504 1008 2016
data_set_maxlen = 504  # one weak 1 * 7 * 24 / 1
print('data_set_maxlen', data_set_maxlen)




def cur_mask_(cur_mask):
    pnumber = cur_mask.shape[0]
    for inx in range(pnumber):
        if int(np.sum(cur_mask[inx]))==0:
            cur_mask[inx][-1]=1.0
    if int(np.sum(cur_mask))==pnumber:
        print('building cur_mask, success')
    else:
        print('building cur_mask, error')
    return cur_mask


# Build readers, discretizers, normalizers
train_reader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                         listfile=os.path.join(args.data, 'train_listfile.csv'), small_part=False)

val_reader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                       listfile=os.path.join(args.data, 'val_listfile.csv'), small_part=False)

test_reader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                        listfile=os.path.join(args.data, 'test_listfile.csv'), small_part=False)


discretizer = Discretizer(timestep=timestep_,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

train_data, normalizer = load_data_utils.dp_load_data(dataloader=train_reader, discretizer=discretizer, normalizer=None)
train_len = np.array([x.shape[0] for x in train_data[0][0]], dtype=int)
max_train = np.max(train_len)
mean_train = np.mean(train_len)
median_train = np.median(train_len)
xx = []
yy = []
y0 = []
for len_ in range(np.min(train_len), max_train+1, 1):
    xx.append(len_)
    yy.append(np.sum(train_len==len_))
    y0.append(0.0)
import matplotlib.pyplot as mplot
mplot.rcParams['figure.figsize'] = (60, 60)
mplot.plot(xx, yy)
mplot.plot(xx, y0)
mplot.show()
val_data = load_data_utils.dp_load_data(dataloader=val_reader, discretizer=discretizer, normalizer=normalizer)
test_data = load_data_utils.dp_load_data(dataloader=test_reader, discretizer=discretizer, normalizer=normalizer)

max_len_train = max([x.shape[0] for x in train_data[0][0]])
max_len_val = max([x.shape[0] for x in val_data[0][0]])
max_len_test = max([x.shape[0] for x in test_data[0][0]])
maxlen = max(max_len_train, max_len_val, max_len_test)
print(max_len_train, max_len_val, max_len_test, maxlen,
      'mean:', np.mean(np.array([x.shape[0] for x in train_data[0][0]])),
      'median:', np.median(np.array([x.shape[0] for x in train_data[0][0]])),
      'mode:', np.argmax(np.bincount(np.array([x.shape[0] for x in train_data[0][0]])))
      )
exit(0)

train_x = train_data[0][0]
train_mask = train_data[0][1]
train_cur_mask = train_data[0][2]
train_dt = [i * timestep_ for i in train_mask]  # xxx minutes
train_y = train_data[1]
print('before')
print('5 example <len, mask, curmask>')
print([ sum(p) for p in train_mask[:5]])
print(train_mask[:5])
print(train_cur_mask[:5])
train_x = np.array(common_utils.pad_sequences(train_x, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
train_mask = np.array(common_utils.pad_sequences(train_mask, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
train_cur_mask = np.array(common_utils.pad_sequences(train_cur_mask, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
train_cur_mask = cur_mask_(train_cur_mask)
train_dt = np.array(common_utils.pad_sequences(train_dt, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
train_y = np.array(common_utils.pad_sequences(train_y, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
print('after')
print('5 example <len, mask, curmask>')
print([ sum(p) for p in train_mask[:5]])
print(train_mask[:5])
print(train_cur_mask[:5])
train_dict = {'X': train_x, 'MASK': train_mask, 'CUR_MASK': train_cur_mask, 'INTERVAL': train_dt, 'Y': train_y, 'NAME': train_data[0][3]}

val_x = val_data[0][0]
val_mask = val_data[0][1]
val_cur_mask = val_data[0][2]
val_dt = [i * timestep_ for i in val_mask]  # xxx minutes
val_y = val_data[1]
val_x = np.array(common_utils.pad_sequences(val_x, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
val_mask = np.array(common_utils.pad_sequences(val_mask, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
val_cur_mask = np.array(common_utils.pad_sequences(val_cur_mask, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
val_cur_mask = cur_mask_(val_cur_mask)
val_dt = np.array(common_utils.pad_sequences(val_dt, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
val_y = np.array(common_utils.pad_sequences(val_y, maxlen=data_set_maxlen, padding='post', truncating='post'),dtype=float)
val_dict = {'X': val_x, 'MASK': val_mask, 'CUR_MASK': val_cur_mask, 'INTERVAL': val_dt, 'Y': val_y, 'NAME': val_data[0][3]}

test_x = test_data[0][0]
test_mask = test_data[0][1]
test_cur_mask = test_data[0][2]
test_dt = [i * timestep_ for i in test_mask]  # xxx minutes
test_y = test_data[1]
test_x = np.array(common_utils.pad_sequences(test_x, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
test_mask = np.array(common_utils.pad_sequences(test_mask, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
test_cur_mask = np.array(common_utils.pad_sequences(test_cur_mask, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
test_cur_mask = cur_mask_(test_cur_mask)
test_dt = np.array(common_utils.pad_sequences(test_dt, maxlen=data_set_maxlen, padding='post', truncating='post'), dtype=float)
test_y = np.array(common_utils.pad_sequences(test_y, maxlen=data_set_maxlen, padding='post', truncating='post'),dtype=float)
test_dict = {'X': test_x, 'MASK': test_mask, 'CUR_MASK': test_cur_mask, 'INTERVAL': test_dt, 'Y': test_y, 'NAME': test_data[0][3]}

print(train_x.shape, train_mask.shape, train_dt.shape, train_y.shape, '\n', 
      val_x.shape, val_mask.shape, val_dt.shape, val_y.shape, '\n',
      test_x.shape, test_mask.shape, test_dt.shape, test_y.shape)

#  mimic3
#  (29030, 168, 76) (29030, 168) (29030, 168) (29030, 168, 1)
#  (6335, 168, 76) (6335, 168) (6335, 168) (6335, 168, 1)
#  (6237, 168, 76) (6237, 168) (6237, 168) (6237, 168, 1)

# pickle_file = open('/home/xxx/ehr_data/dp/dp_mimic3_train_data_{}.pkl'.format(data_set_maxlen), 'wb')
# pickle.dump(train_dict, pickle_file)
# pickle_file.close()
# 
# pickle_file = open('/home/xxx/ehr_data/dp/dp_mimic3_val_data_{}.pkl'.format(data_set_maxlen), 'wb')
# pickle.dump(val_dict, pickle_file)
# pickle_file.close()
# 
# pickle_file = open('/home/xxx/ehr_data/dp/dp_mimic3_test_data_{}.pkl'.format(data_set_maxlen), 'wb')
# pickle.dump(test_dict, pickle_file)
# pickle_file.close()

pickle_file = open('/home/xxx/ehr_data/dp/dp_mimic3_train_data.pkl', 'wb')
pickle.dump(train_dict, pickle_file)
pickle_file.close()

pickle_file = open('/home/xxx/ehr_data/dp/dp_mimic3_val_data.pkl', 'wb')
pickle.dump(val_dict, pickle_file)
pickle_file.close()

pickle_file = open('/home/xxx/ehr_data/dp/dp_mimic3_test_data.pkl', 'wb')
pickle.dump(test_dict, pickle_file)
pickle_file.close()