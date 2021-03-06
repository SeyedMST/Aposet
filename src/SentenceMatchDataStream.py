import numpy as np
import re
import random
import math
import matplotlib
import matplotlib.pyplot as plt

import sys
eps = 1e-8

def make_batches(size, batch_size = 500):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]

def make_batches_as (instances, is_training, batch_size=100000, max_answer_size=100000, equal_box_per_batch = True):

    if equal_box_per_batch == True:
        box_count_per_batch = batch_size // max_answer_size
    else:
        box_count_per_batch = 2000000

    ans = []
    ans_len = []
    question_count = []
    last_size = 0
    ss = 0
    ss_tot = 0
    start = 0
    count = 0
    smaller_than_count = 0
    for x in instances:
        if ss_tot == 0:
            last_size = len(x[1])
        if (len(x[1]) == last_size and ss + len(x[1]) <= batch_size and count +1 <=box_count_per_batch):
            ss += len(x[1])
        else:
            if count < box_count_per_batch:
                smaller_than_count += 1
            ans.append((start, ss_tot))
            ans_len.append(last_size)
            question_count.append(count)
            count = 0
            last_size = len(x[1])
            start = ss_tot
            ss = len (x[1])
        ss_tot += len(x[1])
        count += 1
    ans.append((start, ss_tot))
    ans_len.append(last_size)
    question_count.append(count)
    #if equal_box_per_batch == True:
    #    print ("smaller than count ", smaller_than_count)
    return (ans, question_count, ans_len)

def wikiQaGenerate(filename, is_training, is_ndcg, zero_pad, zero_pad_max = 130):
    data = open(filename, 'rt')
    question_dic = {}
    question_count = 0 #wiki 2,118
    all_count = 0 #wiki 20,360
    del_question_count = 0 #wiki 1,245 (59% of questions deleted, 873 question remaine)
    del_all_count = 0 #wiki 11,688 (57% of pairs deleted, 8,672 remaine(9.9 answer per question))

    max_val = {}
    min_val = {}
    cnt2 = 0
    cnt1 = 0
    for line in data:
        line = line.strip()
        #if line.startswith('-'): continue
        item = re.split(" ", line)
        label = int (item[0])

        # if label == 1:
        #     cnt1 += 1
        #if is_ndcg == False and label == 2:
        #    label = 1
            # cnt2+=1
            # if cnt2 % 10 == 0:
            #     print (cnt1, cnt2)

        if label == 2:
            label = 1
        question = str (re.split(":", item [1])[1])
        input_vector = []
        for i in range (2, len (item)):
            if str(item [i]).startswith('#'): break
            r = float (re.split(":", item [i])[1])
            input_vector.append(r)
            max_val.setdefault(i-2, -100000000.0)
            min_val.setdefault(i-2, 1000000000.0)
            if r > max_val[i-2]:
                max_val [i-2] = r
            if r < min_val[i-2]:
                min_val[i-2] = r
        question_dic.setdefault(question,{})
        question_dic[question].setdefault("question",[])
        question_dic[question].setdefault("answer",[])
        question_dic[question].setdefault("label",[])
        question_dic[question]["question"].append(question)
        question_dic[question]["answer"].append(input_vector)
        question_dic[question]["label"].append(label)
        all_count += 1
        if all_count == 100000:
            break
    for key in list(question_dic):
        question_count += 1
        question_dic[key]["question"] = question_dic[key]["question"]
        question_dic[key]["answer"] = question_dic[key]["answer"]
        # ll = []
        # for x in question_dic[key]['answer']:
        #     l = []
        #     for i in range(len(x)):
        #         if max_val[i] - min_val[i] > eps:
        #             l.append((x[i] - min_val[i])/(max_val[i] - min_val[i]))
        #         else:
        #             l.append(x[i]-min_val[i])
        #     ll.append(l)
        #
        # if len (ll[0]) > 100:
        #     question_dic[key]["answer"] = ll

        last_x = -100
        flag_delete = True

        for x in question_dic[key]["label"]:
            if last_x == -100:
                last_x = x
            if x != last_x:
                flag_delete = False
                break
        if (flag_delete == True):
            del_question_count += 1
            del_all_count += len(question_dic[key]["question"])
            del(question_dic[key])
    #print ("pairs count", all_count - del_all_count)
    question = list()
    answer = list()
    label = list()
    pairs_count = 0
    pos_neg_pair_count = 0
    total_pair_count = 0

    mask = np.ones((len(question_dic), zero_pad_max), np.float32)
    ind = -1
    for item in question_dic.values():
        ind += 1
        label_list = item ['label']
        answer_list = item ['answer']
        if zero_pad == True and is_training == True:

            l = []
            for i in range (len(label_list)):
                l.append((label_list[i], answer_list[i]))
            l = sorted(l, key=lambda instance: (instance[0]),reverse=True)  # sort based on len (answer[i])
            label_list = []
            answer_list = []
            for i in range (len (l)):
                label_list.append(l[i][0])
                answer_list.append(l[i][1])

            last_size = len(label_list)
            new_size = zero_pad_max - len(label_list)
            for i in range (new_size):
                label_list.append(-1)
                mask [ind, i + last_size] = 0

            input_dim = len (answer_list[0])
            for i in range (new_size):
                l =  []
                for j in range (input_dim):
                    l.append(0)
                answer_list.append(l)

        label.append(label_list)
        answer.append(answer_list) # answer[i] = list of answers of question i
        question += [([item["question"][0]])[0]] # question[i] = question i

        pairs_count += len(item ['answer'])

    question = np.array(question) # list of questions
    answer = np.array(answer) # list of list of answers
    label = np.array(label) #list of list of labels

    instances = []
    for i in range(len(question)):
        instances.append((question[i], answer[i], label[i], mask[i]))
    #random.shuffle(instances)  # random works inplace and returns None
    if is_training == True:
        batches = make_batches_as(instances, is_training)
    else:
        batches = make_batches(pairs_count)
    ans = []
    candidate_answer_length = []
    for x in (instances):
        candidate_answer_length.append(len(x[1]))
        for j in range (len(x[1])):
            ans.append(
                (x[0], x[1][j], x[2][j], x[3][j]))
    #print ("Questions: ",len(instances), " pairs: ", len(ans))
    return (ans, batches, candidate_answer_length)


class SentenceMatchDataStream(object):
    def __init__(self, inpath, is_training,
                 isShuffle=False ,isLoop=False, isSort=True, zero_pad = False, is_ndcg = False):

        self.batch_as_len = []
        self.batch_question_count = []
        self.candidate_answer_length = []
        self.real_candidate_answer_length = []

        instances, r, self.candidate_answer_length = wikiQaGenerate(inpath, is_ndcg = is_ndcg,  is_training=is_training,zero_pad=zero_pad)
        if is_training == True:
            batch_spans = r[0]
            self.batch_question_count = r[1]
            self.batch_as_len = r[2]
        else:
            batch_spans = r
            self.batch_question_count = 0
            self.batch_as_len = 0

        self.num_instances = len(instances)
        self.batches = []
        max_answer_len = 0
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            label_id_batch = []
            input_vector_batch = []
            mask_batch = []

            for i in range(batch_start, batch_end):
                (quesion_id, input_vector, label_id, mask) = instances[i]
                label_id_batch.append(label_id)
                input_vector_batch.append(input_vector)
                mask_batch.append(mask)

            if len (label_id_batch) > max_answer_len:
                max_answer_len = len(label_id_batch)
            label_id_batch = np.array(label_id_batch)
            input_vector_batch = np.array(input_vector_batch)
            mask_batch = np.array(mask_batch)

            self.batches.append((label_id_batch, input_vector_batch, mask_batch))

        print ("max_answer_len", max_answer_len) #121 MQ2008
        instances = None
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def answer_count (self, i):
        return self.batch_as_len[i]

    def question_count (self, i):
        return self.batch_question_count[i]

    def real_answer_count (self, i):
        return self.real_candidate_answer_length[i]

    def get_candidate_answer_length (self):
        return self.candidate_answer_length

    def nextBatch(self):
        if self.cur_pointer>=self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0 
            if self.isShuffle: np.random.shuffle(self.index_array)
#         print('{} '.format(self.index_array[self.cur_pointer]))
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch, self.index_array[self.cur_pointer-1]

    def reset(self):
        #if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0
    
    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i>= self.num_batch: return None
        return self.batches[i]
        
