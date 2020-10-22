import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as k

from recordmaker import Seconddataset, Thirddataset, Fourthdataset
from trainandtransfer import ModelAccess


def get_cin_parameters(model):
    # collect "gamma" and "beta" in CRN model
    target_var = []
    fake_input = tf.ones([1, 800, 270])
    fake_pid = tf.ones([1], dtype=tf.int64)
    fake_aid = tf.ones([1], dtype=tf.int64)
    model([fake_input, fake_pid, fake_aid], training=True)
    model_variables = model.trainable_variables
    for i in model_variables:
        if "gamma" in i.name or "beta" in i.name:
            target_var.append(i)
    return target_var


class TransferAccess:
    def __init__(self, action, ckpt_dir, source_folder="D:/datas"):
        ini_set = Thirddataset(source_folder="D:/datas", seen=5, train_num=5)
        # ini_set = dataset(source_folder="D:/datas")

        self.all_set = ini_set.listset
        self.Cmodel = ModelAccess("import", ckpt_dir=ckpt_dir).Cmodel
        self.trainable_var = get_cin_parameters(self.Cmodel)

        '''test_input = tf.ones([1, 800, 270])
        test_pid = tf.ones([1], dtype=tf.int64)
        test_aid = tf.ones([1], dtype=tf.int64)
        model = self.Cmodel
        model([test_input, test_aid, test_pid], training=True)
        a = model.trainable_variables
        self.trainable_var = []
        for i in a:
            if "gamma" in i.name or "beta" in i.name:
                self.trainable_var.append(i)'''

        self.epochs = 100
        self.batch_size = 32
        self.ckpt_dir = ckpt_dir

        self.early_stoping_step = 10

        # initialization of early_stop

        self.optimizer = tf.keras.optimizers.SGD(0.5)
        # self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.checkpoint = tf.train.Checkpoint(optimizer_trans=self.optimizer, contasnet=self.Cmodel)

        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_dir, max_to_keep=5)

        if action == "train":
            i = 0
            while i < self.epochs:
                self.work_on_dataset(self.all_set[0][1], training=True)

                # acc score
                score, _ = self.work_on_dataset(self.all_set[1][1], training=False)
                self.ckpt_manager.save()
                print("Current Epoch is {}, metric score is {},".format(i, score))
                i += 1

        if action == "visualize":
            score, confusion_matrix = self.work_on_dataset(self.all_set[1][1], training=False)
            print(score)
            print(confusion_matrix)

        if action == "test":
            score, confusion_matrix = self.work_on_dataset(self.all_set[2][1], training=False)
            print(score)
            print(confusion_matrix)

        else:
            print("finish load model!")

    def work_on_dataset(self, dset, training):
        # train or eval
        logits_list = []
        label_list = []
        score_list = []

        dset = dset.cache().shuffle(15200, reshuffle_each_iteration=1)
        dset = dset.batch(self.batch_size, drop_remainder=True)

        if training:
            dset = dset.repeat(4)

        for csi, labelp, labela in dset:
            if training:
                with tf.GradientTape() as tape:
                    logits = self.Cmodel([csi, labelp, labela], training=True)

                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labela)
                    loss = k.mean(loss)
                    # print(loss)
                    grd = tape.gradient(loss, self.trainable_var)

                    self.optimizer.apply_gradients(zip(grd, self.trainable_var))

            else:
                logits = self.Cmodel([csi, labelp], training=False)
                acc = np.mean(k.argmax(logits, axis=-1).numpy() == labela.numpy())

                logits_list.append(k.argmax(logits, axis=-1).numpy())
                label_list.append(labela.numpy())
                score_list.append(acc)
        if not training:
            confusion_matrix = tf.math.confusion_matrix(labels=np.array(label_list).reshape([-1]),
                                                        predictions=np.array(logits_list).reshape([-1]))
            return np.mean(score_list), confusion_matrix


if __name__ == '__main__':
    TransferAccess("train", ckpt_dir="C:/Users/620/PycharmProjects/GAN/CrossID/CrossPosition/Ckpt")  # 17:29-19:49
    a = "visualize"
