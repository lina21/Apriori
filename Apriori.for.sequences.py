#import matplotlib.pyplot as plt
#import numpy as np
#import numpy.random as npr
import math
import collections
import copy
import time
from operator import itemgetter
import pickle


def Save(data, file_name):
    with open(file_name, 'wb+') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def Read(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def GetData():
    dataset = []
    file = open("data-2016.csv", "r")
    av = file.readlines()
    file.close()
    for i in range(len(av)):
        dataset.append([])
        string = av[i].split()
        for j in range(len(string)):
            dataset[len(dataset)-1].append(string[j])
    Save(dataset, 'data/data.pickle')
    return dataset

def GetUniqueCourses(dataset):
        courses = dict()
        for i in range(len(dataset)):
                j = 2
                while j < len(dataset[i]):
                        if dataset[i][j] not in courses.keys():
                                courses[int(dataset[i][j])] = copy.deepcopy(dataset[i][j+1])
                        j += 5
        return courses


def MonthDistance(d1, d2):
    y1 = int(d1.split('-')[0])
    m1 = int(d1.split('-')[1])
    y2 = int(d2.split('-')[0])
    m2 = int(d2.split('-')[1])
    return (y2-y1)*12 - m1 + m2 - 1


def GetTransactions(dataset):
        trans = []
        for i in range(len(dataset)):
                j = 2
                courses = []
                prev_timestamp = dataset[i][1]
                while j < len(dataset[i]):
                        itemset = []
                        timestamp = dataset[i][j-1]
                        for l in range(MonthDistance(prev_timestamp, timestamp)):
                            courses.append([])
                        prev_timestamp = timestamp
                        while (j < len(dataset[i])) and (dataset[i][j-1] == timestamp) :
                            itemset.append(int(dataset[i][j]))
                            j += 5
                        courses.append(sorted(itemset))
                trans.append(courses)
        Save(sorted(trans), 'data/transactions.pickle')
        return sorted(trans)


def GetCoursesCodes(dataset):
    return sorted(list(map(int, GetUniqueCourses(dataset).keys())))


def make_data_files():
    dataset = GetData()
    GetTransactions(dataset)
    '''
    try:
        with open('data/data.pickle', 'rb') as f:
            dataset = pickle.load(f)
        with open('data/transactions.pickle', 'rb') as f:
            transactions = pickle.load(f)
    except:
        print('Cannot load Data files!')
        sys.exit(1)
    '''


def SupportCount(transactions, itemset, maxspan):
        support_count = 0
        for i in range(len(transactions)):
            j = 0
            num_matches = 0
            prev_timestamp = 0
            for item in itemset:
                found = False
                while j < (len(transactions[i])):
                    #transactions[i]&item - intersection of 2 lists
                    cur_timestamp = j
                    sub = transactions[i][j]
                    if ((len(set(sub) & set(item))) == len(item)) and (cur_timestamp - prev_timestamp <= maxspan):
                        found = True
                        prev_timestamp = j
                        num_matches += 1
                        j += 1
                        break
                    else:
                        j += 1
                if found == False:
                    break
            if num_matches == len(itemset):
                support_count += 1
        return support_count


def Support(transactions, itemset, maxspan):
        return SupportCount(transactions, itemset, maxspan)/len(transactions)


def Union(itemset1, itemset2):
    return list(sorted(set().union(itemset1, itemset2)))


def GetMergedSeq(s1, s2):
    #s is [[], []] - list of lists
    s2_last_el_index = len(s2)-1
    seq1 = [s1[0][1:]] + s1[1:]
    if seq1[0]  == []:
        seq1.pop(0)
    seq2 = s2[0:s2_last_el_index] + [s2[s2_last_el_index][0:len(s2[s2_last_el_index])-1]]
    if seq2[len(seq2)-1] == []:
        seq2.pop(s2_last_el_index)
    if seq1 == seq2:
        if len(s2[s2_last_el_index]) > 1:
            if s2[s2_last_el_index][len(s2[s2_last_el_index])-2] <= s2[s2_last_el_index][len(s2[s2_last_el_index])-1]:
                return s1[0:len(s1)-1] + [s1[len(s1)-1] + [s2[s2_last_el_index][len(s2[s2_last_el_index])-1]]]
            else:
                return []
        else:
            return s1[0:] + [[s2[s2_last_el_index][len(s2[s2_last_el_index])-1]]]
    else:
        return []


def GetTwoSequences(alist):
    res_list = []
    for i in range(len(alist)):
        for j in range(len(alist)):
            first = alist[i]
            second = alist[j]
            #if subsequences are identical the return possible merged sequences
            res_list.append([first, second])
            if first <= second:
                res_list.append([first + second])
    return res_list


def Generate(lists):
    response = []
    if len(lists) > 1:
        index = 0
        while index < len(lists) - 1:
            first = lists[index]
            index2 = index
            while index2 < len(lists):
                second = lists[index2]
                merged_seq = GetMergedSeq(first, second)
                if merged_seq != []:
                    response.append(merged_seq)
                index2 += 1
            index += 1
    return response


def get_elem_index_number(alist, index):
    count_events = 0
    if index == 0:
        return 0, 0
    else:
        for i in range(len(alist)):
            for j in range(len(alist[i])):
                count_events += 1
                if count_events == index:
                    return i, j


def get_k_minus_1_subsequence(alist, index):
    el, ev = get_elem_index_number(alist, index)
    middle = [alist[el][0:ev] + alist[el][ev+1:]]
    if len(middle[0]) == 0:
        middle = []
    return (alist[0:el] + middle + alist[el+1:])


def AdditionalPruning(previous_lists, lists):
    k_minus_1 = len(previous_lists[0])
    i = 0
    while i < (len(lists)):
    #start considering subsequences, which are obtained from sequence by dropping event j
        pruned = False
        for j in range(0, k_minus_1):
            if get_k_minus_1_subsequence(lists[i], j) not in previous_lists:
                lists.remove(lists[i])
                pruned = True
                break
        if pruned == False:
            i += 1
    return lists


def Apriori(dataset, transactions, min_support, k, maxspan):
    k_minus_1 = 2
    try:
        lists = Read('data/TwoSequencesMinSup'+str(min_support)+'MaxSpan'+str(maxspan)+'.pickle')
        candidates = []
        previous_lists = copy.deepcopy(lists)
    except:
        lists = []
        courses = sorted(GetCoursesCodes(dataset))
        for i in range(len(courses)):
            if Support(transactions, [[courses[i]]], maxspan) > min_support:
                lists.append([courses[i]])
        candidates = GetTwoSequences(lists)
        del lists
        print('2-sequences are generated.')
        #do pruning of candidate 2-sequences
        previous_lists = []
        for i in range(len(candidates)):
            if Support(transactions, candidates[i], maxspan) > min_support:
                previous_lists.append(candidates[i])
        Save(previous_lists, 'data/TwoSequencesMinSup'+str(min_support)+'MaxSpan'+str(maxspan)+'.pickle')
        lists = copy.deepcopy(previous_lists)
    while (len(lists) != 0) and (k_minus_1 < k):
        k_minus_1 += 1
        del candidates
        candidates = Generate(lists)
        print('Current k = ' + str(k_minus_1) + ', number of candidate sequences: ', len(candidates))
        lists = copy.deepcopy(candidates)
        #additional pruning

        if k_minus_1 > 2:
            len_before = len(lists)
            candidates = AdditionalPruning(previous_lists, lists)
            print('First pruning ended. Pruned', str(len_before- len(candidates)), 'candidate sequences')

        del lists
        lists = []
        for i in range(len(candidates)):
            if Support(transactions, candidates[i], maxspan) > min_support:
                lists.append(candidates[i])
        #now we have only frequent candidates
        if len(lists) == 0:
            lists = previous_lists
            break
        else:
            previous_lists = copy.deepcopy(lists)
        print('Second pruning ended. Number of ' + str(k_minus_1) +'-sequences:', len(lists))

    supports = {}
    for i in range(len(lists)):
        sup = Support(transactions, lists[i], maxspan)
        if sup not in supports.keys():
            supports[sup] = []
        supports[sup].append(lists[i])

    return collections.OrderedDict(sorted(supports.items(), reverse=True))


def load_data():
    return Read('data/data.pickle')


def load_transactions():
    return Read('data/transactions.pickle')


def main():
    try:
        dataset = load_data()
        transactions = load_transactions()
    except:
        make_data_files()
        dataset = load_data()
        transactions = load_transactions()

    min_support =  float(input('Enter minimum support of the sequence: '))
    k =  int(input('Enter the length of the sequence (k): '))
    max_span =  int(input('Enter maximum span (in months): '))
    supports = Apriori(dataset, transactions, min_support, k, max_span)

    print('RESULTS:')
    courses = GetUniqueCourses(dataset)
    #translate sequences with codes to sequences with course names
    for sup_seq in supports.items():
        print('Sequences with Support', sup_seq[0])
        for s in range(len(sup_seq[1])):
            seq = sup_seq[1][s]
            seq_level_1 = []
            for i in range(len(seq)):
                seq_level_2 = []
                for j in range(len(seq[i])):
                    t = seq[i][j]
                    seq_level_2.append(courses[seq[i][j]])
                seq_level_1.append(seq_level_2)
            print(seq, seq_level_1)
            del seq_level_1
    del dataset, transactions, courses, supports


if __name__ == "__main__":
           main()