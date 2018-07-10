import tensorflow as tf
#import my_rnn
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn

eps = 1e-8


def get_place_holder (q_count, type, shape):
    ans = ()
    for _ in range(q_count):
        ans += (tf.placeholder(type, shape=shape),)
    return ans


class SentenceMatchModelGraph(object):
    def __init__(self, num_classes=3,
                learning_rate=0.001, optimize_type='adam', lambda_l2=1e-5,
                 is_training=True, input_dim = 46
                 , prediction_mode = 'list_wise'
                 ,loss_type = 'poset_net'
                 , pos_avg = True,
                 q_count=2, dropout_rate = 0.05):

        # ======word representation layer======

        self.truth = get_place_holder (q_count, tf.float32, [None])#q_count*[tf.placeholder(tf.float32, [None])] # [batch_size]
        self.input_vector = get_place_holder (q_count, tf.float32, [None, None])
        loss_list = []
        score_list = []
        with tf.variable_scope ('salamzendegi'):
            for i in range (q_count):
                sec_dim = 1
                if prediction_mode == 'point_wise':
                    sec_dim = num_classes
                v = self.input_vector[i]#tf.reshape(self.input_vector[i], [1, tf.shape(self.input_vector[i])[0]])
                w = tf.get_variable("w", [input_dim, input_dim],dtype=tf.float32)
                b = tf.get_variable("b", [input_dim],dtype=tf.float32)
                v = tf.nn.relu(tf.matmul(v, w) + b)
                if is_training:
                    v = tf.nn.dropout(v, (1 - dropout_rate))
                else:
                    v = tf.multiply(v, (1 - dropout_rate))
                w_1 = tf.get_variable("w_1", [input_dim, sec_dim],dtype=tf.float32)
                b_1 = tf.get_variable("b_1", [sec_dim],dtype=tf.float32)
                logits = tf.matmul(v, w_1) + b_1
                logits = tf.reshape(logits, [-1])
                if prediction_mode != 'point_wise':
                    #score_list.append(tf.reshape(logits, [-1]))
                    self.score = tf.reshape(logits, [-1])
                    #logits = tf.reshape(logits, shape=[self.question_count[i], self.answer_count[i]])
                    #gold_matrix = tf.reshape(self.truth[i], shape=[self.question_count[i], self.answer_count[i]])
                    #g1_matrix = tf.ceil(self.truth[i] - eps)
                    if prediction_mode == 'list_wise':
                        if loss_type == 'list_net':
                            if i == 1:
                                print ('ssss')
                            self.logits = tf.nn.softmax(logits)  # [question_count, answer_count]
                            self.soft_truth = tf.nn.softmax(self.truth[i])
                            #self.soft_truth = tf.divide(self.truth[i], tf.reduce_sum(self.truth[i]))
                            # loss_list.append(tf.reduce_sum(
                            #     tf.multiply(soft_truth, tf.log(soft_truth+eps)) - tf.multiply(soft_truth, tf.log(logits))
                            #    ))
                            self.loss = tf.reduce_sum(
                                tf.multiply(self.soft_truth, tf.log(self.soft_truth))
                                - tf.multiply(self.soft_truth, tf.log(self.logits))
                               )
                        elif loss_type == 'poset_net':
                            gold = self.truth[i]
                            #mask2:
                            g1 = tf.maximum(gold, 1.0)
                            self.mask2 = g1 - 1.0
                            self.mask01 = 1.0 - self.mask2
                            self.mask12 = tf.minimum(gold, 1.0)
                            self.mask1 = self.mask12 - self.mask2
                            self.mask0 = self.mask01 - self.mask1
                            self.fi, pos_cnt = self.poset_loss(logits,self.mask2, self.mask01)
                            self.fi1, pos_cnt1 = self.poset_loss(logits,self.mask1, self.mask0)
                            fi = tf.add(self.fi, self.fi1)
                            pos_cnt =tf.add(pos_cnt, pos_cnt1)
                            if pos_avg == True:
                                fi = tf.divide(fi, pos_cnt)
                            self.loss = fi


                else: #pointwise-cross entropy
                    logit_list = tf.unstack(logits,axis = 1 ,num=2)
                    score_list.append(logit_list[1])
                    gold_matrix = tf.one_hot(self.truth[i], num_classes, dtype=tf.float32)
                    loss_list.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gold_matrix)))
                tf.get_variable_scope().reuse_variables()

        #self.loss = tf.stack (loss_list, 0)
        #self.loss = tf.reduce_mean(self.loss, 0)
        #self.score = tf.concat(score_list, 0)
        if optimize_type == 'adadelta':
            clipper = 50 
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 
        elif optimize_type == 'sgd':
            self.global_step = tf.Variable(0, name='global_step', trainable=False) # Create a variable to track the global step.
            min_lr = 0.000001
            self._lr_rate = tf.maximum(min_lr, tf.train.exponential_decay(learning_rate, self.global_step, 30000, 0.98))
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self._lr_rate).minimize(self.loss)
        elif optimize_type == 'ema':
            tvars = tf.trainable_variables()
            train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            # Create an ExponentialMovingAverage object
            ema = tf.train.ExponentialMovingAverage(decay=0.9999)
            # Create the shadow variables, and add ops to maintain moving averages # of var0 and var1.
            maintain_averages_op = ema.apply(tvars)
            # Create an op that will update the moving averages after each training
            # step.  This is what we will use in place of the usual training op.
            with tf.control_dependencies([train_op]):
                self.train_op = tf.group(maintain_averages_op)
        elif optimize_type == 'adam':
            clipper = 50
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            tvars = tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss = self.loss + lambda_l2 * l2_loss
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars)) 

        extra_train_ops = []
        train_ops = [self.train_op] + extra_train_ops
        self.train_op = tf.group(*train_ops)


    #def list_mle(self, logits):

    def poset_loss(self, logits, pos_mask, neg_mask):
        has_pos = tf.reduce_max(pos_mask)
        has_neg = tf.reduce_max(neg_mask)
        has_pos_neg = tf.multiply(has_pos , has_neg)
        pos_count = tf.reduce_sum(pos_mask)  # [1]
        neg_exp = tf.exp(tf.multiply(neg_mask, logits))  # [a]
        neg_exp = tf.multiply(neg_exp, neg_mask)
        neg_exp_sum = tf.reduce_sum(neg_exp)  # [1]
        pos_exp = tf.exp(tf.multiply(pos_mask, logits))  # [a]
        fi = tf.log(1 + tf.divide(neg_exp_sum, pos_exp))#+ (1.0-has_pos_neg)))  # [a]
        fi = tf.multiply(fi, pos_mask)  # [a]
        fi = tf.reduce_sum(fi)  # [1]
        # if pos_avg == True:
        #     fi = tf.divide(fi, pos_count)  # [1]
        #     loss_list.append(fi)
        # else:
        #     loss_list.append(fi)
        fi = tf.multiply(fi , has_pos_neg)
        pos_count = tf.multiply(pos_count ,has_pos_neg)
        return fi , pos_count

    def get_score(self):
        return self.__score

    def set_score(self, value):
        self.__score = value

    def del_score(self):
        del self.__score

    def get_input_vector(self):
        return self.__input_vector

    def set_input_vector(self, value):
        self.__input_vector = value

    def del_input_vector(self):
        del self.__input_vector


    def get_truth(self):
        return self.__truth


    def get_loss(self):
        return self.__loss


    def get_train_op(self):
        return self.__train_op


    def get_global_step(self):
        return self.__global_step


    def get_lr_rate(self):
        return self.__lr_rate


    def set_truth(self, value):
        self.__truth = value


    def set_loss(self, value):
        self.__loss = value


    def set_train_op(self, value):
        self.__train_op = value


    def set_global_step(self, value):
        self.__global_step = value


    def set_lr_rate(self, value):
        self.__lr_rate = value


    def del_truth(self):
        del self.__truth

    def del_loss(self):
        del self.__loss


    def del_train_op(self):
        del self.__train_op


    def del_global_step(self):
        del self.__global_step


    def del_lr_rate(self):
        del self.__lr_rate

    truth = property(get_truth, set_truth, del_truth, "truth's docstring")
    loss = property(get_loss, set_loss, del_loss, "loss's docstring")
    train_op = property(get_train_op, set_train_op, del_train_op, "train_op's docstring")
    global_step = property(get_global_step, set_global_step, del_global_step, "global_step's docstring")
    lr_rate = property(get_lr_rate, set_lr_rate, del_lr_rate, "lr_rate's docstring")
    score = property(get_score, set_score, del_score, "asdfasdfa")
    input_vector = property(get_input_vector, set_input_vector, del_input_vector, "asdfasdfa")


    