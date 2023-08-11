import keras
import math
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from collections import Counter
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.metrics import average_precision_score
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold

import dataset_utils
import img_tools
import static_values as sv
import tf_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.utils import plot_model

is_model_printed = False

input_size = sv.STATIC_VALUES.image_size[0]
input_channels = 3

epochs = 30
batch_size = sv.STATIC_VALUES.batch_size

n_folds = 5

training = [True, True, True, True, True]

ensemble_voting = False  # If True, use voting for model ensemble, otherwise use averaging

df_train_data = pd.read_csv(sv.STATIC_VALUES.base_dir + '1.local/train.csv')
df_test_data = pd.read_csv(sv.STATIC_VALUES.base_dir + '1.local/test.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = sv.STATIC_VALUES.labels

# create map files
if os.path.exists(sv.STATIC_VALUES.local_dir + 'mapings/label_map.npy'):
    label_map = np.load(sv.STATIC_VALUES.local_dir + 'mapings/label_map.npy').item()
else:
    label_map = {l: i for i, l in enumerate(labels)}
    np.save(sv.STATIC_VALUES.local_dir + 'mapings/label_map.npy', label_map)

if os.path.exists(sv.STATIC_VALUES.local_dir + 'mapings/inv_label_map.npy'):
    inv_label_map = np.load(sv.STATIC_VALUES.local_dir + 'mapings/inv_label_map.npy').item()
else:
    inv_label_map = {i: l for l, i in label_map.items()}
    np.save(sv.STATIC_VALUES.local_dir + 'mapings/inv_label_map.npy', inv_label_map)

if os.path.exists(sv.STATIC_VALUES.local_dir + 'mapings/names_index_train_map.npy'):
    names_index_train_map = np.load(sv.STATIC_VALUES.local_dir + 'mapings/names_index_train_map.npy').item()
else:
    names_index_train_map = {df_train_data.values[i][0]: i for i in range(len(df_train_data))}
    np.save(sv.STATIC_VALUES.local_dir + 'mapings/names_index_train_map.npy', names_index_train_map)

if os.path.exists(sv.STATIC_VALUES.local_dir + 'mapings/names_index_test_map.npy'):
    names_index_test_map = np.load(sv.STATIC_VALUES.local_dir + 'mapings/names_index_test_map.npy').item()
else:
    names_index_test_map = {df_test_data.values[ii][0]: ii for ii in range(len(df_test_data))}
    np.save(sv.STATIC_VALUES.local_dir + 'mapings/names_index_test_map.npy', names_index_test_map)

data_tools = dataset_utils.DatasetOrganizer(df_train_data, df_test_data)

kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)

fold_count = 0

y_full_test = []
thres_sum = np.zeros(sv.STATIC_VALUES.labels_count, np.float32)

# train
for train_index, test_index in kf.split(df_train_data):

    fold_count += 1
    print('Fold ', fold_count)

    df_valid = df_train_data.ix[test_index]
    print('Validating on {} samples'.format(len(df_valid)))


    def valid_generator():
        while True:
            for start in range(0, len(df_valid), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_valid))
                df_valid_batch = df_valid[start:end]
                for f, tags in df_valid_batch.values:
                    # img = img_tools.read_tif(sv.STATIC_VALUES.base_dir + 'train-tif-v2/{}.tif'.format(f))
                    index = names_index_train_map[f]
                    img = data_tools.get_train_image((index))
                    img = img_tools.transformations(img, np.random.randint(6))
                    # img = img_tools.transformations(img, 6)
                    targets = np.zeros(sv.STATIC_VALUES.labels_count)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch


    df_train = df_train_data.ix[train_index]
    if training[fold_count - 1]:
        print('Training on {} samples'.format(len(df_train)))


    def train_generator():
        while True:
            for start in range(0, len(df_train), batch_size):
                x_batch = []
                y_batch = []
                end = min(start + batch_size, len(df_train))
                df_train_batch = df_train[start:end]
                for f, tags in df_train_batch.values:
                    # img = img_tools.read_tif(sv.STATIC_VALUES.base_dir + 'train-tif-v2/{}.tif'.format(f))
                    index = names_index_train_map[f]
                    img = data_tools.get_train_image((index))
                    img = img_tools.transformations(img, np.random.randint(6))
                    # img = img_tools.transformations(img, 6)
                    targets = np.zeros(sv.STATIC_VALUES.labels_count)
                    for t in tags.split(' '):
                        targets[label_map[t]] = 1
                    x_batch.append(img)
                    y_batch.append(targets)
                x_batch = np.array(x_batch, np.float32)
                y_batch = np.array(y_batch, np.uint8)
                yield x_batch, y_batch


    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        images_tensor = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        labels = tf.placeholder(tf.float32, shape=[None, sv.STATIC_VALUES.labels_count])
        learning_rate = tf.placeholder(tf.float32, name='lr')

    output = tf_model.get_model(images_tensor)
    each_error = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output, name='cross_cost')
    error = tf.reduce_mean(each_error, name='mean_cost')

    optimizer = tf.train.AdamOptimizer(learning_rate, name='opt').minimize(error)

    global_step = 0
    init_lr = 1e-3
    decay_rate = 0.96
    decay_steps = 5000
    for epoch in range(10):
        print('epoch: {}'.format((epoch + 1)))

        print('Training...')
        errors = []

        for ii in range(dataset_org.batch_count):
            lr = init_lr * math.pow(decay_rate, (global_step / decay_steps))

            _images, names = dataset_org.next_batch()
            if _images is None:
                break

            _images = np.asarray(_images, dtype=np.float32)
            _images = np.reshape(_images, newshape=(-1, 224, 224, 3))

            _labels = dataset_org.get_labels_all(names)
            _labels = np.reshape(_labels, newshape=(-1, sv.STATIC_VALUES.labels_count))

            feed_dict = {labels: _labels, images_tensor: _images, learning_rate: lr}
            """
            _, error, each, sum, pred = sess.run(
                [model.optimizer, model.error, model.each_error, model.sum, model.predict],
                feed_dict=feed_dict)"""
            _, error, pred, summary_res = sess.run([model.optimizer, model.error, model.predict, summary_cost],
                                                   feed_dict=feed_dict)
            errors.append(error)
            writer.add_summary(summary_res, global_step)
            global_step += 1

    # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    if not is_model_printed:
        model.summary()
        with open("model.txt", 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        plot_model(model, to_file='model.png')
        is_model_printed = not is_model_printed

    callbacks = [EarlyStopping(monitor='val_loss', patience=4, verbose=1, min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=2, verbose=1),
                 ModelCheckpoint(
                     filepath=sv.STATIC_VALUES.local_dir + 'weights/best_weights.fold_' + str(fold_count) + '.hdf5',
                     verbose=1,
                     save_best_only=True,
                     save_weights_only=True),
                 TensorBoard(log_dir=sv.STATIC_VALUES.xception_graph_location, batch_size=20, write_grads=True)]

    if training[fold_count - 1]:
        train_steps_count = ((len(df_train) // batch_size) + 1) if (len(df_train) % batch_size != 0) \
            else (len(df_train) // batch_size)
        validation_steps_count = ((len(df_valid) // batch_size) + 1) if ((len(df_valid) % batch_size) != 0) \
            else (len(df_valid) // batch_size)
        model.fit_generator(generator=train_generator(),
                            steps_per_epoch=train_steps_count,
                            epochs=epochs,
                            verbose=2,
                            callbacks=callbacks,
                            validation_data=valid_generator(),
                            validation_steps=validation_steps_count)


    def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
        def mf(x):
            p2 = np.zeros_like(p)
            for i in range(sv.STATIC_VALUES.labels_count):
                p2[:, i] = (p[:, i] > x[i]).astype(np.int)
            score = fbeta_score(y, p2, beta=2, average='samples')
            return score

        x = [0.2] * sv.STATIC_VALUES.labels_count
        for i in range(sv.STATIC_VALUES.labels_count):
            best_i2 = 0
            best_score = 0
            for i2 in range(resolution):
                i2 /= float(resolution)
                x[i] = i2
                score = mf(x)
                if score > best_score:
                    best_i2 = i2
                    best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)
        return x


    # Load best weights
    model.load_weights(filepath=sv.STATIC_VALUES.local_dir + 'weights/best_weights.fold_' + str(fold_count) + '.hdf5')

    validation_steps_count = ((len(df_valid) // batch_size) + 1) if ((len(df_valid) % batch_size) != 0) \
        else (len(df_valid) // batch_size)
    p_valid = model.predict_generator(generator=valid_generator(),
                                      steps=validation_steps_count)

    y_valid = []
    for f, tags in df_valid.values:
        targets = np.zeros(sv.STATIC_VALUES.labels_count)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        y_valid.append(targets)
    y_valid = np.array(y_valid, np.uint8)

    # TODO This have been changed
    # Find optimal f2 thresholds for local validation set
    thres = optimise_f2_thresholds(y_valid, p_valid, verbose=False)

    print('mAP = {}'.format(average_precision_score(y_valid, np.array(p_valid), average='samples')))

    thres_sum += np.array(thres, np.float32)


    def test_generator(transformation):
        while True:
            for start in range(0, len(df_test_data), batch_size):
                x_batch = []
                end = min(start + batch_size, len(df_test_data))
                df_test_batch = df_test_data[start:end]
                for f in df_test_batch.values[:, 0]:
                    # img = img_tools.read_tif(sv.STATIC_VALUES.base_dir + 'train-tif-v2/{}.tif'.format(f))
                    index = names_index_test_map[f]
                    img = data_tools.get_test_image((index))
                    img = img_tools.transformations(img, transformation)
                    # img = img_tools.transformations(img, 6)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32)
                yield x_batch


    # 6-fold TTA
    p_full_test = []
    test_steps_count = (len(df_test_data) // batch_size) + 1 if ((len(df_test_data) % batch_size) != 0) \
        else (len(df_test_data) // batch_size)

    tta_count = 6
    # tta_count = 1
    for i in range(tta_count):
        print('Calculate prediction for transformation: {}'.format((i + 1)))
        p_test = model.predict_generator(generator=test_generator(transformation=i),
                                         steps=test_steps_count)
        p_full_test.append(p_test)

    p_test = np.array(p_full_test[0])
    for i in range(1, tta_count):
        p_test += np.array(p_full_test[i])
    p_test /= tta_count

    y_full_test.append(p_test)

result = np.array(y_full_test[0])
if ensemble_voting:
    for f in range(len(y_full_test[0])):  # For each file
        for tag in range(sv.STATIC_VALUES.labels_count):  # For each tag
            preds = []
            for fold in range(n_folds):  # For each fold
                preds.append(y_full_test[fold][f][tag])
            pred = Counter(preds).most_common(1)[0][0]  # Most common tag prediction among folds
            result[f][tag] = pred
else:
    for fold in range(1, n_folds):
        result += np.array(y_full_test[fold])
    result /= n_folds

for i, tag in enumerate(sv.STATIC_VALUES.labels):
    df_test_data[tag] = result[:, i]
df_test_data.to_csv('submission.csv', index=False)
print('Submission Created!')
