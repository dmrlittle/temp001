import argparse

import pandas as pd

from multiprocessing import Pool
from collections import defaultdict
import os

from nltk.tag import pos_tag

from utils import get_related_forms, dump_json_set, load_json_set, clean_title
from utils import get_variants_and_derivatives, get_syn_thesaurus_net, get_syn_thesaurus_com
from utils import get_column_variants_per_pos, get_describe, get_related
from utils import get_common_words, load_models
from utils import get_ant_thesaurus_com, get_ant_antonymfor
from database import open_session, initialize_db
from database import ColumnD, ColumnI, ColumnJ, ColumnK, ColumnL, ColumnN
from tqdm import tqdm


MODELS = load_models()

def task_2():
    if os.path.exists('output/task_2.json'):
        print("Task 2 data found...")
        return load_json_set('output/task_2.json')
    else:
        print("Loading data for Task 2...")
        df = pd.read_excel('DATASET.xlsx', sheet_name='Task 2')
        set1 = set(map(lambda x: x.strip(), df['Unnamed: 1'].dropna().values))
        print("Getting related forms...")
        nouns, adjs, advs, vrbs = get_related_forms(set1, should_ban=True)
        print(len(nouns), len(adjs), len(advs), len(vrbs))
        with Pool(5) as p:
            f = p.map(get_related_forms, [nouns, adjs, advs, vrbs])

        for (nouns_, adjs_, advs_, vrbs_) in f:
            nouns = nouns.union(nouns_)
            adjs = adjs.union(adjs_)
            advs = advs.union(advs_)
            vrbs = vrbs.union(vrbs_)
        print(len(nouns), len(adjs), len(advs), len(vrbs))
        res = {'n': nouns, 'a': adjs, 'r': advs, 'v': vrbs}

        dump_json_set(res, 'output/task_2_just_wn.json')
    return res


def task_3(word, th_k=0.7, th_l=0.6):
    print("Executing task 3...")

    # column d
    column_d = get_variants_and_derivatives(word)
    
    nouns = column_d['n']
    adjs = column_d['a']
    advs = column_d['r']
    verbs = column_d['v']

    # COLUMN E & F
    res_prm = defaultdict(list)
    res_snd = defaultdict(list)

    # iterate over nouns
    for n_ in nouns:
        res1_top, res1_bot = get_syn_thesaurus_com(n_)
        res2_top, res2_bot = get_syn_thesaurus_net(n_)
        
        for pos_tag_ in {'a', 'n', 'r', 'v'}:
            res_prm[pos_tag_] += res1_top[pos_tag_] + res2_top[pos_tag_] #+ [word]
            res_snd[pos_tag_] += res1_bot[pos_tag_] + res2_bot[pos_tag_]
        res_prm['n'] += [n_]
    
    # iterate over adjetives
    for a_ in adjs:
        res1_top, res1_bot = get_syn_thesaurus_com(a_)
        res2_top, res2_bot = get_syn_thesaurus_net(a_)

        for pos_tag_ in {'a', 'n', 'r', 'v'}:
            res_prm[pos_tag_] += res1_top[pos_tag_] + res2_top[pos_tag_] #+ [word]
            res_snd[pos_tag_] += res1_bot[pos_tag_] + res2_bot[pos_tag_]
        res_prm['a'] += [a_]
    
    # iterate over adverbs
    for r_ in advs:
        res1_top, res1_bot = get_syn_thesaurus_com(r_)
        res2_top, res2_bot = get_syn_thesaurus_net(r_)
        
        for pos_tag_ in {'a', 'n', 'r', 'v'}:
            res_prm[pos_tag_] += res1_top[pos_tag_] + res2_top[pos_tag_] #+ [word]
            res_snd[pos_tag_] += res1_bot[pos_tag_] + res2_bot[pos_tag_]
        res_prm['r'] += [r_]
    
    # iterate over verbs
    for v_ in verbs:
        res1_top, res1_bot = get_syn_thesaurus_com(v_)
        res2_top, res2_bot = get_syn_thesaurus_net(v_)
        
        for pos_tag_ in {'a', 'n', 'r', 'v'}:
            res_prm[pos_tag_] += res1_top[pos_tag_] + res2_top[pos_tag_] #+ [word]
            res_snd[pos_tag_] += res1_bot[pos_tag_] + res2_bot[pos_tag_]
        res_prm['v'] += [v_]

    related_words = {'primary': {k: set(v) for k, v in res_prm.items()},
                     'secondary': {k: set(v) for k, v in res_snd.items()}}

    # column i
    column_i = get_describe(word)
    # column j
    column_j = get_related(word)

    # column K
    column_k = get_common_words(word, related_words, models=MODELS, threshold=th_k)
    # related forms of k
    aux_1_k = get_column_variants_per_pos(column_k['primary'])
    # cutoff
    aux_2_k = get_common_words(word, {'primary': aux_1_k, 'secondary': defaultdict(set)}, models=MODELS, threshold=th_l)

    # column l: add d and k back again
    column_l_prim = defaultdict(set)
    for pos in {'a', 'v', 'r', 'n'}:
        column_l_prim[pos] = column_d[pos].union(*[column_k['primary'][pos], aux_2_k['primary'][pos]])

    column_l = {'primary': column_l_prim,
                'remainder': aux_2_k['remainder']}

    ants = set()
    for elem_ in nouns.union(*[adjs, advs, verbs]):
        ants = ants.union(*[get_ant_thesaurus_com(elem_), get_ant_antonymfor(elem_)])

    nns = set()
    vbs = set()
    adj = set()
    adv = set()

    for ant_ in ants:
        aux_ = get_variants_and_derivatives(ant_)
        nns = nns.union(aux_['n'])
        vbs = vbs.union(aux_['v'])
        adj = adj.union(aux_['a'])
        adv = adv.union(aux_['r'])
        ant_pos = pos_tag([ant_])[0][0]
        if ant_pos.startswith('NN'):
            nns.add(ant_)
        elif ant_pos.startswith('JJ'):
            adj.add(ant_)
        elif ant_pos.startswith('VB') or ant_pos == 'MD':
            vbs.add(ant_)
        elif ant_pos.startswith('RB'):
            adv.add(ant_)
    column_n = {'n': nns, 'v': vbs, 'a': adj, 'r': adv}

    print("Task 3 completed.")
    return column_d, column_i, column_j, column_k, column_l, column_n


def task_4(word, th_k=0.7, th_l=0.4):
    print("Executing task 4...")
    
    syns_net = get_syn_thesaurus_net(word, is_task_4=True)
    syns_com = get_syn_thesaurus_com(word, is_task_4=True)

    # column e
    column_e = defaultdict(set)
    for pos in {'a', 'r', 'n', 'v'}:
        column_e[pos] = syns_net[pos].union(syns_com[pos])

    # column f
    column_f = column_e.copy()
    nouns = column_e['n']
    advs = column_e['r']
    adjs = column_e['a']
    verbs = column_e['v']

    #N = None
    N = 10
    print('A')
    for idx, nn in tqdm(enumerate(list(nouns)[:N], start=1), total=N):
        # thesaurus net
        aux = get_syn_thesaurus_net(nn, is_task_4=True)

        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)

        # thesaurus com
        aux = get_syn_thesaurus_com(nn, is_task_4=True)

        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)

        column_f['n'].add(nn)
    print('B')
    for idx, vb in tqdm(enumerate(list(verbs)[:N], start=1), total=N):
        # thesaurus net
        aux = get_syn_thesaurus_net(vb, is_task_4=True)
        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)
        # thesaurus com
        aux = get_syn_thesaurus_com(vb, is_task_4=True)
        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)
        column_f['v'].add(vb)
    print('C')
    for idx, ad in tqdm(enumerate(list(adjs)[:N], start=1), total=N):
        # thesaurus net
        aux = get_syn_thesaurus_net(ad, is_task_4=True)
        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)
        # thesaurus com
        aux = get_syn_thesaurus_com(ad, is_task_4=True)
        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)
        column_f['a'].add(ad)
    print('D')
    for idx, ad in tqdm(enumerate(list(advs)[:N], start=1), total=N):
        # thesaurus net
        aux = get_syn_thesaurus_net(ad, is_task_4=True)
        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)
        # thesaurus com
        aux = get_syn_thesaurus_com(ad, is_task_4=True)
        for pos, elems in aux.items():
            column_f[pos] = column_f[pos].union(elems)
        column_f['r'].add(ad)
    # column g
    column_g = {'v': column_f['v']}

    # column k
    column_k = get_common_words(word, {'primary': column_g, 'secondary': defaultdict(set)}, models=MODELS, threshold=th_k, pos_tags_=column_g.keys())

    # related forms of k
    aux_1_k = get_column_variants_per_pos(column_k['primary'])['v']
    # cutoff
    aux_2_k = get_common_words(word, {'primary': {'v': aux_1_k}, 'secondary': defaultdict(set)}, models=MODELS, threshold=th_l, pos_tags_={'v'})

    # column l
    column_l_prim = defaultdict(set)
    for pos in aux_2_k['primary'].keys():
        column_l_prim[pos] = column_k['primary'][pos].union(aux_2_k['primary'][pos])

    column_l = {'primary': column_l_prim,
                'remainder': aux_2_k['remainder']}

    print("Task 4 completed.")
    return column_k, column_l

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=None, help='just firs n elements')
    parser.add_argument('-db', type=str, default='database/database.db', help='where to store the database')
    args = parser.parse_args()
    n = args.n
    db = args.db
    initialize_db(db_path=db)
    
    session = open_session(db_path=db)

    df = pd.read_excel('DATASET.xlsx', sheet_name='Task 3 - Input')
    dataset = df[['Sr', 'Title']]

    print(f'Total Number of Processing to do : {dataset.shape[0]}')
    
    try:
        n = int(input('Limit process to > '))
        if n<1 :
            n=None
    except:
        n=None

    column_c = defaultdict(dict)
    d = dict()

    if os.path.exists('processed_words.txt'):
        with open('processed_words.txt') as fp:
            processed = set([x.strip() for x in fp])
    else:
        processed = set()

    task_2_words = task_2()
    for col_ in ['d', 'i', 'j', 'k', 'l', 'n']:
        with open(f'output/task_3_column_{col_}.csv', 'w') as fp:
            pass
    for col_ in ['k', 'l']:
        with open(f'output/task_4_column_{col_}.csv', 'w') as fp:
            pass

    for _, tag_data in dataset[1:n].iterrows():
        tag = tag_data['Title']
        idx = tag_data['Sr']
        try:
            idx = int(idx)
        except:
            pass

        print(f'Processing entry: {idx}...\n')
        try:
            # column c
            tokens = clean_title(tag)
            for t in tokens:
                s = t[0]
                column_c[s].update({idx: set(tokens)})
        except Exception as e:
            print(e)
            continue
        for token in tokens:
            if token in processed:
                continue
            processed.add(token)

            col_d_t3, col_i_t3, col_j_t3, col_k_t3, col_l_t3, col_n_t3 = task_3(token)

            # POPULATE COLUMN D
            with open('output/task_3_column_d.csv', 'a') as fp1:
                for pos_, words in col_d_t3.items():
                    for w_ in words:
                        r = {'task': 3,
                             'tag_id': str(idx),
                             'tag': tag.strip(),
                             'root': token,
                             'pos': pos_,
                             'word': w_}
                        session.add(ColumnD(**r))
                        fp1.write(','.join([str(idx), tag.strip(), token, pos_, w_]))
                        fp1.write('\n')

            # POPULATE COLUMN I
            with open('output/task_3_column_i.csv', 'a') as fp1:
                for w_ in col_i_t3:
                    r = {'task': 3,
                         'tag_id': str(idx),
                         'tag': tag.strip(),
                         'root': token,
                         'word': w_}
                    session.add(ColumnI(**r))
                    fp1.write(','.join([str(idx), tag.strip(), token, w_]))
                    fp1.write('\n')

            # POPULATE COLUMN J
            with open('output/task_3_column_j.csv', 'a') as fp1:
                for w_ in col_j_t3:
                    r = {'task': 3,
                         'tag_id': str(idx),
                         'tag': tag.strip(),
                         'root': token,
                         'word': w_}
                    session.add(ColumnJ(**r))
                    fp1.write(','.join([str(idx), tag.strip(), token, w_]))
                    fp1.write('\n')

            # POPULATE COLUMN K
            with open('output/task_3_column_k.csv', 'a') as fp1:
                for cat_, dicts in col_k_t3.items():
                    for pos_, words in dicts.items():
                        for w_ in words:
                            r = {'task': 3,
                                 'tag_id': str(idx),
                                 'tag': tag.strip(),
                                 'root': token,
                                 'cat': cat_,
                                 'pos': pos_,
                                 'word': w_}
                            session.add(ColumnK(**r))
                            fp1.write(','.join([str(idx), tag.strip(), token, cat_, pos_, w_]))
                            fp1.write('\n')

            # POPULATE COLUMN L
            with open('output/task_3_column_l.csv', 'a') as fp1:
                for cat_, dicts in col_l_t3.items():
                    for pos_, words in dicts.items():
                        for w_ in words:
                            r = {'task': 3,
                                 'tag_id': str(idx),
                                 'tag': tag.strip(),
                                 'root': token,
                                 'cat': cat_,
                                 'pos': pos_,
                                 'word': w_}
                            session.add(ColumnL(**r))
                            fp1.write(','.join([str(idx), tag.strip(), token, cat_, pos_, w_]))
                            fp1.write('\n')

            # POPULATE COLUMN N
            with open('output/task_3_column_n.csv', 'a') as fp1:
                for pos_, words in col_n_t3.items():
                    for w_ in words:
                        r = {'task': 3,
                             'tag_id': str(idx),
                             'tag': tag.strip(),
                             'root': token,
                             'pos': pos_,
                             'word': w_}
                        session.add(ColumnN(**r))
                        fp1.write(','.join([str(idx), tag.strip(), token, pos_, w_]))
                        fp1.write('\n')
            try:
                session.commit()
            except Exception as e:
                print(e)
                session.rollback()

            # TASK 4
            if token in task_2_words['a'] or token in task_2_words['r'] or token in task_2_words['v']:
                continue

            col_k_t4, col_l_t4 = task_4(token)
            with open('output/task_4_column_k.csv', 'a') as fp2:
                for cat_, dicts in col_k_t4.items():
                    for pos_, words in dicts.items():
                        for w_ in words:
                            r = {'task': 4,
                                 'tag_id': str(idx),
                                 'tag': tag.strip(),
                                 'root': token,
                                 'cat': cat_,
                                 'pos': pos_,
                                 'word': w_}
                            session.add(ColumnK(**r))
                            fp2.write(','.join([str(idx), tag.strip(), token, cat_, pos_, w_]))
                            fp2.write('\n')
            with open('output/task_4_column_l.csv', 'a') as fp2:
                for cat_, dicts in col_l_t4.items():
                    for pos_, words in dicts.items():
                        for w_ in words:
                            r = {'task': 4,
                                 'tag_id': str(idx),
                                 'tag': tag.strip(),
                                 'root': token,
                                 'cat': cat_,
                                 'pos': pos_,
                                 'word': w_}
                            session.add(ColumnL(**r))
                            fp2.write(','.join([str(idx), tag.strip(), token, cat_, pos_, w_]))
                            fp2.write('\n')
            try:
                session.commit()
            except Exception as e:
                print(e)
                session.rollback()
    dump_json_set(column_c, 'output/column_c.json')

    with open('processed_words.txt', 'a') as fp:
        for word_ in processed:
            fp.write(word_)
            #fp.write()
