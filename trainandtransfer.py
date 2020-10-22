import os
import numpy as np
import tensorflow as tf
from recordmaker import Seconddataset, Thirddataset, Fourthdataset
from CrossIDnet import CRN
from tensorflow.python.keras import backend as k


class ModelAccess(object):
    def __init__(self, action, ckpt_dir, source_folder="D:/datas"):
        super(ModelAccess, self).__init__()
        assert action in ["train", "import", "visualize", "test"], "Not in action set"
        ini_set = Thirddataset(source_folder=source_folder, seen=5, train_num=5)
        # ini_set = dataset(source_folder=source_folder)

        # 0,0: source_train; 0,1: target_train;1,0: source_dev; 1,1: target_dev;2,0: source_test; 2,1: target_test

        self.all_set = ini_set.listset

        self.Cmodel = CRN()
        os.makedirs(ckpt_dir, exist_ok=True)
        self.epochs = 10
        self.batch_size = 32
        self.ckpt_dir = ckpt_dir

        self.early_stoping_step = 10

        # initialization of early_stop

        self.current_epoch = tf.Variable(0, trainable=False)

        # self.optimizer = tf.compat.v1.train.MomentumOptimizer(5e-3, momentum=.9)
        self.optimizer = tf.keras.optimizers.RMSprop(1e-3)

        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              contasnet=self.Cmodel)

        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, self.ckpt_dir, max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")
        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)

        if action == "train":
            while self.current_epoch.numpy() < self.epochs:
                self.work_on_dataset(self.all_set[0][0], training=True)

                # CV score
                score1, _ = self.work_on_dataset(self.all_set[1][0], training=False)
                score2, _ = self.work_on_dataset(self.all_set[1][1], training=False)
                self.ckpt_manager.save()
                print("Current Epoch is {}, acc score is {}***{},".format(self.current_epoch.numpy(), score1, score2))
                self.current_epoch.assign_add(1)
                # current_gamma = CIN().gamma_mat
        if action == "visualize":
            score, confusion_matrix = self.work_on_dataset(self.all_set[1][0], training=False)
            print(score)
            print(confusion_matrix)

        if action == "test":
            score, confusion_matrix = self.work_on_dataset(self.all_set[2][0], training=False)
            print(score)
            print(confusion_matrix)

        else:
            print("finish load model!")

    def work_on_dataset(self, dset, training):
        # train or eval
        logits_list = []
        label_list = []
        score_list = []

        dset = dset.cache().shuffle(10000, reshuffle_each_iteration=1)
        dset = dset.repeat(10).batch(self.batch_size, drop_remainder=True)

        for csi, labelp, labela in dset:

            if training:
                with tf.GradientTape() as tape:
                    logits = self.Cmodel([csi, labelp, labela], training=True)
                    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labela)
                    loss = k.mean(loss)
                    # print(loss)
                    grd = tape.gradient(loss, self.Cmodel.trainable_variables)
                    self.optimizer.apply_gradients(zip(grd, self.Cmodel.trainable_variables))

            else:
                logits = self.Cmodel([csi, labelp], training=False)
                acc = np.mean(k.argmax(logits, axis=-1).numpy() == labela.numpy())
                # print(acc)
                score_list.append(acc)
                logits_list.append(k.argmax(logits, axis=-1).numpy())
                label_list.append(labela.numpy())

        if not training:
            confusion_matrix = tf.math.confusion_matrix(labels=np.array(label_list).reshape([-1]),
                                                        predictions=np.array(logits_list).reshape([-1]))
            return np.mean(score_list), confusion_matrix


if __name__ == '__main__':
    ModelAccess("train", ckpt_dir="C:/Users/620/PycharmProjects/GAN/CrossID/CrossPosition/Ckpt")
