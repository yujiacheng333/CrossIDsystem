import tensorflow as tf
import numpy as np
import os
import random

# sourcepath = """D:\Party\data\onedata"""
pname = ['DX', 'WX', 'WYY', 'YN', 'YY', 'WB']
act = ['O', 'PO', 'UP', 'X']
# pos = [2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23]


pos = [2, 3, 4, 5, 8, 9, 10, 11]


def add_noise(signal_input):
    # 给数据加指定SNR的高斯噪声
    snr = -15  # -15,-10, -5, 0, 5
    mu = 0

    signal_noise = np.zeros_like(signal_input)
    for i in range(signal_input.shape[1]):
        signal_power = np.linalg.norm(signal_input[:, i]) ** 2 / signal_input[:, i].size  # 此处是信号的std**2
        noise_variance = signal_power / np.power(10, (snr / 10))  # 此处是噪声的std**2
        sigma = np.sqrt(noise_variance)
        signal_noise[:, i] = signal_input[:, i] + random.gauss(mu, sigma)
    return signal_noise


class Pdataset(object):

    def __init__(self, source_folder, seen, train_num, reset=False):
        # seen = 4
        # train_num = 5
        # len(pos) = 4/8//12/16
        source_folder = source_folder + "/pname/" + str(len(pos)) + "/" + str(seen) + "/" + str(train_num) + "/"
        os.makedirs(source_folder, exist_ok=True)

        self.reset = reset
        self.seen = seen
        # self.unseen = 6 - self.seen
        reshuffle_pos = np.random.choice(np.arange(len(pos)), replace=False, size=len(pos))
        reshuffle_index = np.random.choice(np.arange(50), replace=False, size=50)
        if reset:
            # haha = input("1?")
            writer_train_seen = tf.io.TFRecordWriter(source_folder + "/source_train.tfrecord")
            writer_dev_seen = tf.io.TFRecordWriter(source_folder + "/source_dev.tfrecord")
            writer_test_seen = tf.io.TFRecordWriter(source_folder + "/source_test.tfrecord")
            writer_train_unseen = tf.io.TFRecordWriter(source_folder + "/target_train.tfrecord")
            writer_dev_unseen = tf.io.TFRecordWriter(source_folder + "/target_dev.tfrecord")
            writer_test_unseen = tf.io.TFRecordWriter(source_folder + "/target_test.tfrecord")

            all_counter = 0

            for cs, ON in enumerate(pname):
                for ca, OA in enumerate(act):
                    for p in reshuffle_pos:
                        all_counter += 1
                        print(all_counter)
                        pos_index = pos[p]
                        counter = 0
                        for index in reshuffle_index:

                            local_fp = "D:/Party/data/onedata/{}/{}/{}-{}{}-{}.npy".format(ON, OA, OA, ON, pos_index,
                                                                                           index)
                            local_data = np.load(local_fp)
                            if cs < self.seen:
                                # seen:
                                if counter < 40:
                                    writer_train_seen.write(self._serialize_example(local_data, cs, ca))
                                elif counter < 45:
                                    writer_dev_seen.write(self._serialize_example(local_data, cs, ca))
                                else:
                                    writer_test_seen.write(self._serialize_example(local_data, cs, ca))
                            else:
                                # unseen:
                                if counter < train_num:
                                    writer_train_unseen.write(self._serialize_example(local_data, cs, ca))
                                elif counter < 5 + train_num:
                                    writer_dev_unseen.write(self._serialize_example(local_data, cs, ca))
                                else:
                                    writer_test_unseen.write(self._serialize_example(local_data, cs, ca))

                            counter += 1
                            # print(counter)

            writer_train_seen.close()
            writer_dev_seen.close()
            writer_test_seen.close()
            writer_train_unseen.close()
            writer_test_unseen.close()
            writer_dev_unseen.close()
            print("***********************", self.seen, "*****", train_num, "****************************")

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_train.tfrecord")
        self.source_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_dev.tfrecord")
        self.source_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_test.tfrecord")
        self.source_test = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_train.tfrecord")
        self.target_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_dev.tfrecord")
        self.target_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_test.tfrecord")
        self.target_test = raw_dataset.map(self._extract_fn)

        self.listset = [[self.source_train, self.target_train],
                        [self.source_dev, self.target_dev],
                        [self.source_test, self.target_test]]
        del raw_dataset

    def _extract_fn(self, data_record):
        features = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'labelp': tf.io.FixedLenFeature([], tf.int64),
            'labela': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(sample["csi"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [800, 270])
        spk_label = sample["labelp"]
        length = sample["labela"]
        return recv_sig, spk_label, length

    def _serialize_example(self, local_data, labelp, labela):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        signals = local_data.astype(np.float32).reshape([-1])
        feature = {
            'csi': _bytes_feature(signals.tobytes()),
            'labelp': _int64_feature(np.int64(labelp)),
            'labela': _int64_feature(np.int64(labela))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


class Ddataset(object):

    def __init__(self, source_folder, seen, train_num, reset=False):
        # seen = 4
        # train_num = 5
        # len(pos) = 4/8//12/16
        source_folder = source_folder + "/duizhao/" + str(len(pos)) + "/" + str(seen) + "/" + str(train_num) + "/"
        os.makedirs(source_folder, exist_ok=True)

        self.reset = reset
        self.seen = seen
        # self.unseen = 6 - self.seen
        # reshuffle_pos = np.random.choice(np.arange(16), replace=False, size=16)
        reshuffle_pos = np.random.choice(np.arange(len(pos)), replace=False, size=len(pos))
        reshuffle_index = np.random.choice(np.arange(50), replace=False, size=50)
        if reset:
            # haha = input("1?")
            writer_train_seen = tf.io.TFRecordWriter(source_folder + "/source_train.tfrecord")
            writer_dev_seen = tf.io.TFRecordWriter(source_folder + "/source_dev.tfrecord")
            writer_test_seen = tf.io.TFRecordWriter(source_folder + "/source_test.tfrecord")
            writer_train_unseen = tf.io.TFRecordWriter(source_folder + "/target_train.tfrecord")
            writer_dev_unseen = tf.io.TFRecordWriter(source_folder + "/target_dev.tfrecord")
            writer_test_unseen = tf.io.TFRecordWriter(source_folder + "/target_test.tfrecord")

            all_counter = 0

            for cs, ON in enumerate(pname):
                for ca, OA in enumerate(act):
                    for p in reshuffle_pos:
                        all_counter += 1
                        print(all_counter)
                        pos_index = pos[p]
                        counter = 0
                        for index in reshuffle_index:

                            local_fp = "D:/Party/data/onedata/{}/{}/{}-{}{}-{}.npy".format(ON, OA, OA, ON, pos_index,
                                                                                           index)
                            local_data = np.load(local_fp)
                            # add noise fun
                            # local_data = get_noise(local_data)
                            # print(local_fp)
                            if cs + 1 != self.seen:
                                # seen:
                                if counter < 40:
                                    writer_train_seen.write(self._serialize_example(local_data, cs, ca))
                                elif counter < 45:
                                    writer_dev_seen.write(self._serialize_example(local_data, cs, ca))
                                else:
                                    writer_test_seen.write(self._serialize_example(local_data, cs, ca))
                            else:
                                # unseen:
                                if counter < train_num:
                                    writer_train_unseen.write(self._serialize_example(local_data, cs, ca))
                                elif counter < 5 + train_num:
                                    writer_dev_unseen.write(self._serialize_example(local_data, cs, ca))
                                else:
                                    writer_test_unseen.write(self._serialize_example(local_data, cs, ca))

                            counter += 1
                            # print(counter)

            writer_train_seen.close()
            writer_dev_seen.close()
            writer_test_seen.close()
            writer_train_unseen.close()
            writer_test_unseen.close()
            writer_dev_unseen.close()
            print("***********************", self.seen, "*****", train_num, "****************************")

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_train.tfrecord")
        self.source_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_dev.tfrecord")
        self.source_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_test.tfrecord")
        self.source_test = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_train.tfrecord")
        self.target_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_dev.tfrecord")
        self.target_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_test.tfrecord")
        self.target_test = raw_dataset.map(self._extract_fn)

        self.listset = [[self.source_train, self.target_train],
                        [self.source_dev, self.target_dev],
                        [self.source_test, self.target_test]]
        del raw_dataset

    def _extract_fn(self, data_record):
        features = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'labelp': tf.io.FixedLenFeature([], tf.int64),
            'labela': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(sample["csi"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [800, 270])
        # recv_sig = add_noise(recv_sig.numpy())  # add noise function
        spk_label = sample["labelp"]
        length = sample["labela"]
        return recv_sig, spk_label, length

    def _serialize_example(self, local_data, labelp, labela):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        signals = local_data.astype(np.float32).reshape([-1])
        feature = {
            'csi': _bytes_feature(signals.tobytes()),
            'labelp': _int64_feature(np.int64(labelp)),
            'labela': _int64_feature(np.int64(labela))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


class Seconddataset(object):  # 同时迁移人（4->2）和位置（2，5，20，23）

    def __init__(self, source_folder, seen, train_num, reset=False):
        # seen = 4
        # train_num = 5
        # len(pos) = 4/8//12/16
        source_folder = source_folder + "/weizhi1/" + str(len(pos)) + "/" + str(seen) + "/" + str(train_num) + "/"
        os.makedirs(source_folder, exist_ok=True)

        self.reset = reset
        self.seen = seen
        reshuffle_pos = np.random.choice(np.arange(len(pos)), replace=False, size=len(pos))
        reshuffle_index = np.random.choice(np.arange(50), replace=False, size=50)
        if reset:
            # haha = input("1?")
            writer_train_seen = tf.io.TFRecordWriter(source_folder + "/source_train.tfrecord")
            writer_dev_seen = tf.io.TFRecordWriter(source_folder + "/source_dev.tfrecord")
            writer_test_seen = tf.io.TFRecordWriter(source_folder + "/source_test.tfrecord")
            writer_train_unseen = tf.io.TFRecordWriter(source_folder + "/target_train.tfrecord")
            writer_dev_unseen = tf.io.TFRecordWriter(source_folder + "/target_dev.tfrecord")
            writer_test_unseen = tf.io.TFRecordWriter(source_folder + "/target_test.tfrecord")

            all_counter = 0

            for cs, ON in enumerate(pname):
                for ca, OA in enumerate(act):
                    for p in reshuffle_pos:
                        all_counter += 1
                        print(all_counter)
                        pos_index = pos[p]
                        counter = 0
                        for index in reshuffle_index:

                            local_fp = "D:/Party/data/onedata/{}/{}/{}-{}{}-{}.npy".format(ON, OA, OA, ON, pos_index,
                                                                                           index)
                            local_data = np.load(local_fp)
                            if cs < self.seen and pos_index in [2, 5, 20, 23]:
                                # seen:
                                if counter < 40:
                                    writer_train_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                elif counter < 45:
                                    writer_dev_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                else:
                                    writer_test_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                            else:
                                # unseen:
                                if counter < train_num:
                                    writer_train_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                elif counter < 5 + train_num:
                                    writer_dev_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                else:
                                    writer_test_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))

                            counter += 1
                            # print(counter)

            writer_train_seen.close()
            writer_dev_seen.close()
            writer_test_seen.close()
            writer_train_unseen.close()
            writer_test_unseen.close()
            writer_dev_unseen.close()
            print("***********************", self.seen, "*****", train_num, "****************************")

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_train.tfrecord")
        self.source_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_dev.tfrecord")
        self.source_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_test.tfrecord")
        self.source_test = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_train.tfrecord")
        self.target_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_dev.tfrecord")
        self.target_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_test.tfrecord")
        self.target_test = raw_dataset.map(self._extract_fn)

        self.listset = [[self.source_train, self.target_train],
                        [self.source_dev, self.target_dev],
                        [self.source_test, self.target_test]]
        del raw_dataset

    def _extract_fn(self, data_record):
        features = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'labelp': tf.io.FixedLenFeature([], tf.int64),
            'labela': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(sample["csi"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [800, 270])
        spk_label = sample["labelp"]
        length = sample["labela"]
        return recv_sig, spk_label, length

    def _serialize_example(self, local_data, labelp, labela):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        signals = local_data.astype(np.float32).reshape([-1])
        feature = {
            'csi': _bytes_feature(signals.tobytes()),
            'labelp': _int64_feature(np.int64(labelp)),
            'labela': _int64_feature(np.int64(labela))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


class Thirddataset(object):  # 同时迁移人（4->2）和位置（9，10，15，16）

    def __init__(self, source_folder, seen, train_num, reset=False):
        # seen = 4
        # train_num = 5
        # len(pos) = 4/8//12/16
        source_folder = source_folder + "/weizhi2/" + str(len(pos)) + "/" + str(seen) + "/" + str(train_num) + "/"
        os.makedirs(source_folder, exist_ok=True)

        self.reset = reset
        self.seen = seen
        reshuffle_pos = np.random.choice(np.arange(len(pos)), replace=False, size=len(pos))
        reshuffle_index = np.random.choice(np.arange(50), replace=False, size=50)
        if reset:
            # haha = input("1?")
            writer_train_seen = tf.io.TFRecordWriter(source_folder + "/source_train.tfrecord")
            writer_dev_seen = tf.io.TFRecordWriter(source_folder + "/source_dev.tfrecord")
            writer_test_seen = tf.io.TFRecordWriter(source_folder + "/source_test.tfrecord")
            writer_train_unseen = tf.io.TFRecordWriter(source_folder + "/target_train.tfrecord")
            writer_dev_unseen = tf.io.TFRecordWriter(source_folder + "/target_dev.tfrecord")
            writer_test_unseen = tf.io.TFRecordWriter(source_folder + "/target_test.tfrecord")

            all_counter = 0

            for cs, ON in enumerate(pname):
                for ca, OA in enumerate(act):
                    for p in reshuffle_pos:
                        all_counter += 1
                        print(all_counter)
                        pos_index = pos[p]
                        counter = 0
                        for index in reshuffle_index:

                            local_fp = "D:/Party/data/onedata/{}/{}/{}-{}{}-{}.npy".format(ON, OA, OA, ON, pos_index,
                                                                                           index)
                            local_data = np.load(local_fp)
                            if cs < self.seen and pos_index in [9, 10, 15, 16]:
                                # seen:
                                if counter < 40:
                                    writer_train_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                elif counter < 45:
                                    writer_dev_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                else:
                                    writer_test_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                            else:
                                # unseen:
                                if counter < train_num:
                                    writer_train_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                elif counter < 5 + train_num:
                                    writer_dev_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                else:
                                    writer_test_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))

                            counter += 1
                            # print(counter)

            writer_train_seen.close()
            writer_dev_seen.close()
            writer_test_seen.close()
            writer_train_unseen.close()
            writer_test_unseen.close()
            writer_dev_unseen.close()
            print("***********************", self.seen, "*****", train_num, "****************************")

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_train.tfrecord")
        self.source_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_dev.tfrecord")
        self.source_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_test.tfrecord")
        self.source_test = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_train.tfrecord")
        self.target_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_dev.tfrecord")
        self.target_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_test.tfrecord")
        self.target_test = raw_dataset.map(self._extract_fn)

        self.listset = [[self.source_train, self.target_train],
                        [self.source_dev, self.target_dev],
                        [self.source_test, self.target_test]]
        del raw_dataset

    def _extract_fn(self, data_record):
        features = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'labelp': tf.io.FixedLenFeature([], tf.int64),
            'labela': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(sample["csi"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [800, 270])
        spk_label = sample["labelp"]
        length = sample["labela"]
        return recv_sig, spk_label, length

    def _serialize_example(self, local_data, labelp, labela):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        signals = local_data.astype(np.float32).reshape([-1])
        feature = {
            'csi': _bytes_feature(signals.tobytes()),
            'labelp': _int64_feature(np.int64(labelp)),
            'labela': _int64_feature(np.int64(labela))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


class Fourthdataset(object):  # 同时迁移人（WB->other）和位置（9->other）

    def __init__(self, source_folder, seen, train_num, reset=False):
        # seen = 4
        # train_num = 5
        # len(pos) = 4/8//12/16
        source_folder = source_folder + "/weizhi3/" + str(len(pos)) + "/" + str(seen) + "/" + str(train_num) + "/"
        os.makedirs(source_folder, exist_ok=True)

        self.reset = reset
        self.seen = seen
        reshuffle_pos = np.random.choice(np.arange(len(pos)), replace=False, size=len(pos))
        reshuffle_index = np.random.choice(np.arange(50), replace=False, size=50)
        if reset:
            # haha = input("1?")
            writer_train_seen = tf.io.TFRecordWriter(source_folder + "/source_train.tfrecord")
            writer_dev_seen = tf.io.TFRecordWriter(source_folder + "/source_dev.tfrecord")
            writer_test_seen = tf.io.TFRecordWriter(source_folder + "/source_test.tfrecord")
            writer_train_unseen = tf.io.TFRecordWriter(source_folder + "/target_train.tfrecord")
            writer_dev_unseen = tf.io.TFRecordWriter(source_folder + "/target_dev.tfrecord")
            writer_test_unseen = tf.io.TFRecordWriter(source_folder + "/target_test.tfrecord")

            all_counter = 0

            for cs, ON in enumerate(pname):
                for ca, OA in enumerate(act):
                    for p in reshuffle_pos:
                        all_counter += 1
                        print(all_counter)
                        pos_index = pos[p]
                        counter = 0
                        for index in reshuffle_index:

                            local_fp = "D:/Party/data/onedata/{}/{}/{}-{}{}-{}.npy".format(ON, OA, OA, ON, pos_index,
                                                                                           index)
                            local_data = np.load(local_fp)
                            if cs == 5 and pos_index == 9:
                                # seen:
                                if counter < 40:
                                    writer_train_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                elif counter < 45:
                                    writer_dev_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                else:
                                    writer_test_seen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                            else:
                                # unseen:
                                if counter < train_num:
                                    writer_train_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                elif counter < 5 + train_num:
                                    writer_dev_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))
                                else:
                                    writer_test_unseen.write(self._serialize_example(local_data, cs * 16 + p, ca))

                            counter += 1
                            # print(counter)

            writer_train_seen.close()
            writer_dev_seen.close()
            writer_test_seen.close()
            writer_train_unseen.close()
            writer_test_unseen.close()
            writer_dev_unseen.close()
            print("***********************", self.seen, "*****", train_num, "****************************")

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_train.tfrecord")
        self.source_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_dev.tfrecord")
        self.source_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_test.tfrecord")
        self.source_test = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_train.tfrecord")
        self.target_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_dev.tfrecord")
        self.target_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_test.tfrecord")
        self.target_test = raw_dataset.map(self._extract_fn)

        self.listset = [[self.source_train, self.target_train],
                        [self.source_dev, self.target_dev],
                        [self.source_test, self.target_test]]
        del raw_dataset

    def _extract_fn(self, data_record):
        features = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'labelp': tf.io.FixedLenFeature([], tf.int64),
            'labela': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(sample["csi"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [800, 270])
        spk_label = sample["labelp"]
        length = sample["labela"]
        return recv_sig, spk_label, length

    def _serialize_example(self, local_data, labelp, labela):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        signals = local_data.astype(np.float32).reshape([-1])
        feature = {
            'csi': _bytes_feature(signals.tobytes()),
            'labelp': _int64_feature(np.int64(labelp)),
            'labela': _int64_feature(np.int64(labela))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


class dataset(object):
    """train .8 devlp .1 test .1
    seen pname: 4 unseen :2
    """

    def __init__(self, source_folder, reset=False):
        os.makedirs(source_folder, exist_ok=True)
        self.reset = reset
        reshuffle_index = np.random.choice(np.arange(200), replace=False, size=200)
        if reset:
            haha = input("你妈还在吗")
            writer_train_seen = tf.io.TFRecordWriter(source_folder + "/" + "source" + "_train.tfrecord")
            writer_dev_seen = tf.io.TFRecordWriter(source_folder + "/" + "source" + "_dev.tfrecord")
            writer_test_seen = tf.io.TFRecordWriter(source_folder + "/" + "source" + "_test.tfrecord")
            writer_train_unseen = tf.io.TFRecordWriter(source_folder + "/" + "target" + "_train.tfrecord")
            writer_dev_unseen = tf.io.TFRecordWriter(source_folder + "/" + "target" + "_dev.tfrecord")
            writer_test_unseen = tf.io.TFRecordWriter(source_folder + "/" + "target" + "_test.tfrecord")

            for cs, spk in enumerate(pname):
                for ca, a in enumerate(act):
                    counter = 0
                    local_fp = "D:/Party/data/onedata/{}/{}".format(spk, a)
                    for k in np.asarray(os.listdir(local_fp))[reshuffle_index]:
                        fname = os.path.join(local_fp, k)
                        local_data = np.load(fname)

                        if counter < 160:
                            if cs > 3:
                                writer_train_unseen.write(self._serialize_example(local_data, cs, ca))
                            else:
                                writer_train_seen.write(self._serialize_example(local_data, cs, ca))
                        elif counter < 180:
                            if cs > 3:
                                writer_dev_unseen.write(self._serialize_example(local_data, cs, ca))
                            else:
                                writer_dev_seen.write(self._serialize_example(local_data, cs, ca))
                        else:
                            if cs > 3:
                                writer_test_unseen.write(self._serialize_example(local_data, cs, ca))
                            else:
                                writer_test_seen.write(self._serialize_example(local_data, cs, ca))
                        counter += 1
                        print(counter)
            writer_train_seen.close()
            writer_dev_seen.close()
            writer_test_seen.close()
            writer_train_unseen.close()
            writer_test_unseen.close()
            writer_dev_unseen.close()

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_train.tfrecord")
        self.source_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_dev.tfrecord")
        self.source_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/source_test.tfrecord")
        self.source_test = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_train.tfrecord")
        self.target_train = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_dev.tfrecord")
        self.target_dev = raw_dataset.map(self._extract_fn)

        raw_dataset = tf.data.TFRecordDataset(source_folder + "/target_test.tfrecord")
        self.target_test = raw_dataset.map(self._extract_fn)

        self.listset = [[self.source_train, self.target_train],
                        [self.source_dev, self.target_dev],
                        [self.source_test, self.target_test]]
        del raw_dataset

    def _extract_fn(self, data_record):
        features = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'labelp': tf.io.FixedLenFeature([], tf.int64),
            'labela': tf.io.FixedLenFeature([], tf.int64),
        }
        sample = tf.io.parse_single_example(data_record, features)
        recv_sig = tf.io.decode_raw(sample["csi"], tf.float32)
        recv_sig = tf.reshape(recv_sig, [800, 270])
        spk_label = sample["labelp"]
        length = sample["labela"]
        return recv_sig, spk_label, length

    def _serialize_example(self, local_data, labelp, labela):
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            """Returns an int64_list from a bool / enum / int / uint."""
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        signals = local_data.astype(np.float32).reshape([-1])
        feature = {
            'csi': _bytes_feature(signals.tostring()),
            'labelp': _int64_feature(np.int64(labelp)),
            'labela': _int64_feature(np.int64(labela))
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


if __name__ == '__main__':
    '''
    for i in a.source_train:
        print(i[2].numpy())'''
    """a = Seconddataset(source_folder="D:/datas", seen=5, train_num=5, reset=True)
    b = Thirddataset(source_folder="D:/datas", seen=5, train_num=5, reset=True)
    c = Fourthdataset(source_folder="D:/datas", seen=5, train_num=5, reset=True)"""
    '''c = Ddataset(source_folder="D:/datas", seen=3, train_num=5, reset=True)
    d = Ddataset(source_folder="D:/datas", seen=4, train_num=5, reset=True)
    e = Ddataset(source_folder="D:/datas", seen=5, train_num=5, reset=True)
    f = Ddataset(source_folder="D:/datas", seen=6, train_num=5, reset=True)'''

    # done !
    print("done!")
    a = Fourthdataset(source_folder="D:/datas", seen=5, train_num=5, reset=True)
    for i in a.target_dev:
        print(i[1].numpy())

