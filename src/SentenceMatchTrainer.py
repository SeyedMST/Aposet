# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import sys
import time
import re
import tensorflow as tf
#import pandas as pd
import subprocess
import random
import numpy as np

from SentenceMatchDataStream import SentenceMatchDataStream
from SentenceMatchModelGraph import SentenceMatchModelGraph
import namespace_utils


import nltk
eps = 1e-8
FLAGS = None

def evaluate(dataStream, valid_graph, sess, is_ndcg,
            flag_valid = False
             ,first_on_best_model = False, top_k=5):
    dataStream.reset()
    scores = []
    labels = []
    for batch_index in range(dataStream.get_num_batch()):
        cur_dev_batch = dataStream.get_batch(batch_index)
        (label_id_batch, input_vector_batch) = cur_dev_batch
        feed_dict = {
                    valid_graph.get_truth(): label_id_batch,
                    valid_graph.get_input_vector(): input_vector_batch
                }
        scores.append(sess.run(valid_graph.get_score(), feed_dict=feed_dict))
        labels.append (label_id_batch)
        # if flag_valid == True or first_on_best_model == True:
        #     sent1s.append(sent1_batch)
        #     sent2s.append(sent2_batch)

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    # if flag_valid == True or first_on_best_model == True:
    #     sent1s = np.concatenate(sent1s)
    #     sent2s = np.concatenate(sent2s)
    #return MAP_MRR(scores, labels, dataStream.get_candidate_answer_length(), flag_valid
    #                           ,first_on_best_model)

    if is_ndcg == True:
        return ndcg_k(scores, labels, dataStream.get_candidate_answer_length(), k=top_k, method=1) \
        , ndcg_k(scores, labels, dataStream.get_candidate_answer_length(), k=top_k, method=0)
    else:
        return MAP_MRR(scores, labels, dataStream.get_candidate_answer_length(), flag_valid
                               ,first_on_best_model)


def dcg_at_k(r, k, method):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    print ("hi")
    return 0.

def ndcg_k (logit, gold, candidate_answer_length, k, method=0):
    visited = 0
    c_1 = 0.0
    for i in range(len(candidate_answer_length)):
        prob = logit[visited: visited + candidate_answer_length[i]]
        label = gold[visited: visited + candidate_answer_length[i]]
        visited += candidate_answer_length[i]
        rank_index = np.argsort(prob).tolist()
        rank_index = list(reversed(rank_index))
        r = []
        for x in rank_index:
            r.append(label [x])
        # k1 = k
        # if k1 < len (r):
        #     k1 = len (r)
        dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
        if dcg_max == 0:
            c_1 += 0
        else:
            c_1 += dcg_at_k(r, k, method) / dcg_max
    return c_1 / len (candidate_answer_length)

def MAP_MRR(logit, gold, candidate_answer_length, flag_valid
            ,first_on_best_model):
    c_1_j = 0.0 #map
    c_2_j = 0.0 #mrr
    visited = 0
    #output_sentences = []
    for i in range(len(candidate_answer_length)):
        prob = logit[visited: visited + candidate_answer_length[i]]
        label = gold[visited: visited + candidate_answer_length[i]]
        # if flag_valid == True or first_on_best_model == True:
        #     question = sent1s[visited: visited + candidate_answer_length[i]]
        #     answers = sent2s[visited: visited + candidate_answer_length[i]]
        #     if FLAGS.store_att == True and first_on_best_model == False:
        #         attention_weights = atts [visited: visited + candidate_answer_length[i]]
        visited += candidate_answer_length[i]
        rank_index = np.argsort(prob).tolist()
        rank_index = list(reversed(rank_index))
        score = 0.0
        count = 0.0
        for i in range(1, len(prob) + 1):
            if label[rank_index[i - 1]] > eps:
                count += 1
                score += count / i
        for i in range(1, len(prob) + 1):
            if label[rank_index[i - 1]] > eps:
                c_2_j += 1 / float(i)
                break
        if count > eps:
            c_1_j += score / count

        # if flag_valid == True:
        #     output_sentences.append(word_vocab.to_word_in_sequence(question[0]) + "\n")
        #     for jj in range(len(answers)):
        #         output_sentences.append(str(label[rank_index[jj]]) + " " + str(prob[rank_index[jj]]) + "- " +
        #                                 word_vocab.to_word_in_sequence(answers[rank_index[jj]]) + "\n")
        #     output_sentences.append("AP: {} \n\n".format(score/count))
        # if first_on_best_model == True:
        #     for jj in range(len(answers)):
        #         lj = int(np.ceil(label[rank_index[jj]] - eps))
        #         output_sentences.append(word_vocab.to_word_in_sequence(question[0]) + '\t' +
        #                                 word_vocab.to_word_in_sequence(answers[rank_index[jj]]) + '\t' +
        #                                 str(lj) + '\n')

    my_map = c_1_j/len(candidate_answer_length)
    my_mrr = c_2_j/len(candidate_answer_length)
    return my_map, my_mrr
    # if flag_valid == False and first_on_best_model == False:
    #     return (my_map,my_mrr)
    # else:
    #     return (my_map, my_mrr, output_sentences, output_attention_weights)

# def ouput_prob1(probs, label_vocab, lable_true):
#     out_string = ""
#     for i in range(probs.size):
#         if label_vocab.getWord(i) == lable_true:
#             return probs[i]
#
# def output_probs(probs, label_vocab):
#     out_string = ""
#     for i in range(probs.size):
#         out_string += " {}:{}".format(label_vocab.getWord(i), probs[i])
#     return out_string.strip()


def Get_Next_box_size (index):
    if  (index > FLAGS.end_batch):
        return False
    list = ['1','1','1','2','2','2','3','3','3','4','4','4','5','5','5'] #ndcg1 [kl,pos_avg=True,pos_avg=False] ndcg@10
                                                                        # map1 label 1->0 , 2->1 [listnet: crossentropy]
                                                                        # ndcg3 [kl, pos_avg=True, pos_avg=False] ndcg@1
                                                                        # map4 2->1
                                                                        # map7 2->1 [same ndcg1, 2layer NN]
                                                                        # map8 2->1 [same map7, 1layer NN]
                                                                        # map9 same map7


    #list = ['1', '2', '3', '4', '5'] #ndcg2 [list-netcross entropy]
                                        #map2 [list-net cross entropy T/sum(T) instead of softmax] 1->0 2->1
                                        #map3 [list-net kl-div T/sum(T)] 1->0 2->1
                                        #map5 [list-net cross T/sum(T)] 2->1
                                        #map6 same as map5 but delete test with no correct answers
    FLAGS.end_batch = len(list) -1
    FLAGS.fold = list[index]
    qa_path = 'MQ2008/Fold' + FLAGS.fold + '/'
    FLAGS.train_path = '../data/' +qa_path +'train.txt'
    FLAGS.dev_path= '../data/' + qa_path +'vali.txt'
    FLAGS.test_path= '../data/'+qa_path+'test.txt'
    FLAGS.prediction_mode = 'list_wise'
    FLAGS.iter_count = 15
    FLAGS.max_epochs = 20
    FLAGS.is_ndcg = False
    FLAGS.loss_type = 'list_net'
    if index%3 == 0:
        FLAGS.loss_type = 'list_net' #'list_net' , 'poset_net'
    if index%3 ==1:
        FLAGS.loss_type = 'poset_net'
        FLAGS.pos_avg = True
    if index%3 == 2:
        FLAGS.loss_type = 'poset_net'
        FLAGS.pos_avg = False

    return True


def main(_):

    FLAGS.run_id = 'map9'
    log_dir = FLAGS.model_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    path_prefix = log_dir + "/SentenceMatch.{}".format(FLAGS.suffix)

    namespace_utils.save_namespace(FLAGS, path_prefix + ".config.json")
    output_res_file = open('../result/' + FLAGS.run_id, 'wt')
    while (Get_Next_box_size(FLAGS.start_batch) == True):
        train_path = FLAGS.train_path
        dev_path = FLAGS.dev_path
        test_path = FLAGS.test_path
        best_path = path_prefix + '.best.model'
        trainDataStream = SentenceMatchDataStream(train_path, isShuffle=True, isLoop=True, isSort=True)

        #train_testDataStream = SentenceMatchDataStream(train_path, isShuffle=False, isLoop=True, isSort=True)

        testDataStream = SentenceMatchDataStream(test_path, isShuffle=False, isLoop=True, isSort=True)

        devDataStream = SentenceMatchDataStream(dev_path, isShuffle=False, isLoop=True, isSort=True)

        #print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
        #print('Number of instances in devDataStream: {}'.format(devDataStream.get_num_instance()))
        #print('Number of instances in testDataStream: {}'.format(testDataStream.get_num_instance()))
        # print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
        # print('Number of batches in devDataStream: {}'.format(devDataStream.get_num_batch()))
        # print('Number of batches in testDataStream: {}'.format(testDataStream.get_num_batch()))

        sys.stdout.flush()
        output_res_index = 1
        # best_test_acc = 0
        max_test_ndcg = 0
        # max_dev_ndcg = 0
        while output_res_index <= FLAGS.iter_count:
            # st_cuda = ''
            ssst = FLAGS.run_id
            ssst += str(FLAGS.start_batch)
            # output_res_file = open('../result/' + ssst + '.'+ st_cuda + str(output_res_index), 'wt')
            # output_sentence_file = open('../result/' + ssst + '.'+ st_cuda + str(output_res_index) + "S", 'wt')
            # output_train_file = open('../result/' + ssst + '.'+ st_cuda + str(output_res_index) + "T", 'wt')
            # output_sentences = []
            output_res_index += 1
            # output_res_file.write(str(FLAGS) + '\n\n')
            # stt = str (FLAGS)
            # best_dev_acc = 0.0
            max_test_ndcg_iter = 0
            init_scale = 0.01
            with tf.Graph().as_default():
                initializer = tf.random_uniform_initializer(-init_scale, init_scale)
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    train_graph = SentenceMatchModelGraph(num_classes=3, is_training=True, learning_rate=FLAGS.learning_rate
                                                          ,lambda_l2=FLAGS.lambda_l2, prediction_mode=FLAGS.prediction_mode,
                                                          q_count=FLAGS.question_count_per_batch, loss_type = FLAGS.loss_type,
                                                          pos_avg=FLAGS.pos_avg)
                    tf.summary.scalar("Training Loss", train_graph.get_loss()) # Add a scalar summary for the snapshot loss.

        #         with tf.name_scope("Valid"):
                with tf.variable_scope("Model", reuse=True, initializer=initializer):
                    valid_graph = SentenceMatchModelGraph(num_classes=3, is_training=True, learning_rate=FLAGS.learning_rate
                                                          ,lambda_l2=FLAGS.lambda_l2, prediction_mode=FLAGS.prediction_mode,
                                                          q_count=1, loss_type = FLAGS.loss_type,
                                                          pos_avg=FLAGS.pos_avg)


                initializer = tf.global_variables_initializer()
                vars_ = {}
                #for var in tf.all_variables():
                for var in tf.global_variables():
                    vars_[var.name.split(":")[0]] = var
                saver = tf.train.Saver(vars_)

                with tf.Session() as sess:
                    sess.run(initializer)

                    train_size = trainDataStream.get_num_batch()
                    max_steps = (train_size * FLAGS.max_epochs) // FLAGS.question_count_per_batch
                    epoch_size = max_steps // (FLAGS.max_epochs) + 1
                    total_loss = 0.0
                    start_time = time.time()

                    max_valid = 0
                    for step in range(max_steps):
                        # read data
                        _truth = []
                        _input_vector = []
                        for i in range (FLAGS.question_count_per_batch):
                            cur_batch, batch_index = trainDataStream.nextBatch()
                            (label_id_batch, input_vector_batch) = cur_batch

                            _truth.append(label_id_batch)
                            _input_vector.append(input_vector_batch)

                        #print (_truth)
                        feed_dict = {
                                train_graph.get_truth() : tuple(_truth),
                                train_graph.get_input_vector() : tuple(_input_vector),
                                 }
                        _, loss_value = sess.run([train_graph.get_train_op(), train_graph.get_loss()], feed_dict=feed_dict)
                        #print (loss_value)
                        #print (sess.run([train_graph.truth, train_graph.mask01, train_graph.mask0], feed_dict=feed_dict))
                        #loss_value = sess.run([train_graph.logits1], feed_dict=feed_dict)
                        total_loss += loss_value
                        if (step+1) % epoch_size == 0 or (step + 1) == max_steps:
                            #print(total_loss)
                            duration = time.time() - start_time
                            start_time = time.time()
                            total_loss = 0.0

                            my_map, my_mrr = evaluate(devDataStream, valid_graph, sess ,is_ndcg=FLAGS.is_ndcg)
                            flag_valid = False
                            if my_map > max_valid:
                                max_valid = my_map
                                flag_valid = True
                            if flag_valid == True:
                                my_map, my_mrr = evaluate(testDataStream, valid_graph, sess, is_ndcg=FLAGS.is_ndcg,
                                      flag_valid=flag_valid)
                                if my_map > max_test_ndcg and FLAGS.store_best == True:
                                    best_test_acc = my_map
                                    saver.save(sess, best_path)
                                if my_map > max_test_ndcg:
                                    max_test_ndcg = my_map
                                if my_map > max_test_ndcg_iter:
                                    max_test_ndcg_iter = my_map

            print ("{}-{}: {}".format(FLAGS.start_batch, output_res_index-1, max_test_ndcg_iter))


        print ("fold: {} index: {} - test: {}".format(FLAGS.fold, FLAGS.start_batch, max_test_ndcg))
        output_res_file.write("fold: {} index: {} - test: {}\n".format(FLAGS.fold, FLAGS.start_batch, max_test_ndcg))
        FLAGS.start_batch += FLAGS.step_batch

    output_res_file.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum epochs for training.')
    parser.add_argument('--store_best',default=False, type = bool, help='do we have cuda visible devices?')


    parser.add_argument('--loss_type', default='poset_net', help='do we have cuda visible devices?')
    parser.add_argument('--iter_count', type=int, default=30, help='Maximum epochs for training.')
    parser.add_argument('--fold', type=int, default=1, help='Maximum epochs for training.')


    parser.add_argument('--start_batch', type=int, default=0, help='Maximum epochs for training.')
    parser.add_argument('--end_batch', type=int, default=0, help='Maximum epochs for training.')
    parser.add_argument('--step_batch', type=int, default=1, help='Maximum epochs for training.')

    parser.add_argument('--is_ndcg',default=True, type= bool, help='do we have cuda visible devices?')

    parser.add_argument('--pos_avg',default=True, type= bool, help='do we have cuda visible devices?')
    parser.add_argument('--cross_validate',default=True, type= bool, help='do we have cuda visible devices?')

    parser.add_argument('--question_count_per_batch', type=int, default= 1, help='Number of instances in each batch.')

    qa_path = 'MQ2008/Fold2/'
    parser.add_argument('--optimize_type', type=str, default='adam', help='Optimizer type.')
    parser.add_argument('--prediction_mode', default='list_wise', help = 'point_wise, list_wise, hinge_wise .'                                                                          'point wise is only used for non answer selection tasks')
    parser.add_argument('--train_path', type=str,default = '../data/' +qa_path +'train.txt', help='Path to the train set.')
    parser.add_argument('--dev_path', type=str, default = '../data/' + qa_path +'vali.txt', help='Path to the dev set.')
    parser.add_argument('--test_path', type=str, default = '../data/'+qa_path+'test.txt',help='Path to the test set.')
    parser.add_argument('--model_dir', type=str,default = '../models',help='Directory to save model files.')

    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate.')
    parser.add_argument('--lambda_l2', type=float, default=0.0001, help='The coefficient of L2 regularizer.')
    parser.add_argument('--suffix', type=str, default='normal', required=False, help='Suffix of the model name.')
    parser.add_argument('--run_id', default='10' , help = 'run_id')

    #     print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

