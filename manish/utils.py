import time
import re
from base64 import b64encode, b64decode
from json import dumps, loads, JSONEncoder, dump, load
import pickle
from collections import defaultdict
from bs4 import BeautifulSoup
import requests
from itertools import chain, combinations

from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

#from selenium.webdriver.chrome.options import Options
#chrome_options = Options()
#chrome_options.add_argument('--headless')
#expath = '/home/giovanni/upwork/shawn/notebooks/chromedriver'

    #from selenium.webdriver.firefox.options import Options
    #firefox_options = Options()
    #firefox_options.headless = True
expath = '/var/www/html/collabmedia/taggingsys/geckodriver'
#expath = '/content/taggingsys -Upwork/darshan/geckodriver'

from nltk.tag import pos_tag as pos_tag_
from word_forms.word_forms import get_word_forms
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from words import badwords

def load_models():
    print("Loading concept model...")

    with open('models/concept.pickle', 'rb') as fp:
        model_concept = pickle.load(fp)

    print("Loading fasttext model...")

    with open('models/fast.pickle', 'rb') as fp:
        model_fasttext = pickle.load(fp)

    print("Loading glove model...")

    with open('models/wiki-gigaword.pickle', 'rb') as fp:
        model_glove = pickle.load(fp)

    print("Loading google model...")

    with open('models/google.pickle', 'rb') as fp:
        model_google = pickle.load(fp)

    return {'concept': model_concept, 'google': model_google, 'glove': model_glove, 'fttx': model_fasttext}

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(obj)
        return {'_python_object': b64encode(pickle.dumps(obj)).decode('utf-8')}

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(b64decode(dct['_python_object'].encode('utf-8')))
    return dct


def load_json_set(fpath):
    with open(fpath) as fp:
        return loads(load(fp) ,object_hook=as_python_object)

def dump_json_set(data, fpath):
    with open(fpath, 'w') as fp:
        j = dumps(data, cls=PythonObjectEncoder)
        dump(j, fp)

def get_matches(d, ws):
    def powerset(iterable):
        return chain.from_iterable(combinations(iterable, r) for r in range(1, len(iterable) + 1))
    res = {}
    combs = [e for e in powerset(ws)]

    for comb in combs:
        s = comb[0][0]
        for k, v in d.get(s, {}).items():
            aux = set(comb).intersection(v)
            if aux:
                res[k] = aux
    # change to x[0] for getting just id
    return [x for x in sorted(res.items(), key=lambda x: -(len(x[1])))]


# JUST FOR TASK 2
def get_related_forms(ws, should_ban=False):

    banned_words = set()

    nouns = set()
    adjs = set()
    advs = set()
    verbs = set()

    for idx, w in enumerate(ws, start=1):
        if w in banned_words and should_ban:
            continue

        # word net words
        wn_ws = get_word_forms(w)

        nns = wn_ws['n']
        adj = wn_ws['a']
        adv = wn_ws['r']
        vrb = wn_ws['v']

        candidates = nns.union(*[adv, adj, vrb])

        nouns = nouns.union(nns)
        adjs = adjs.union(adj)
        advs = advs.union(adv)
        verbs = verbs.union(vrb)

        if w in candidates:
            candidates.remove(w)

        for cand in candidates:
            if cand in ws:
                banned_words.add(cand)

    return nouns, adjs, advs, verbs


def clean_title(title, ws_={}):
    aux = re.sub(r'[^A-Za-z]', ' ', title)
    for bw in badwords:
        token = r"\b{}\b".format(bw)
        if re.findall(token, aux):
            aux = re.sub(token, " ", aux)
    aux = aux.split()
    return [word.lower() for word in aux if word not in stop_words and word not in ws_]


def get_variants_and_derivatives(word):
    return get_word_forms(word)


#############
# TASK 3 b)
#############
def get_syn_thesaurus_com(w, is_task_4=False):
    def get_info(driver):
        primary = defaultdict(list)
        secondary = defaultdict(list)
        # primary synonyms
        class_a = 'css-gkae64 etbu2a31'
        # secondary synonyms
        class_b = 'css-q7ic04 etbu2a31'
        class_c = 'css-zu4egz etbu2a31'
        class_d = 'css-1rm5oe2 etbu2a30'

        pos_tag_class = "css-1twju98 e9i53te7"
        soup = BeautifulSoup(driver.page_source, 'lxml')
        pos = soup.find('a', class_=pos_tag_class)
        pos_tag = None
        if pos is None:
            return primary, secondary
        if pos.text.startswith('verb'):
            pos_tag = 'v'
        elif pos.text.startswith('noun'):
            pos_tag = 'n'
        elif pos.text.startswith('adv'):
            pos_tag = 'r'
        elif pos.text.startswith('adj'):
            pos_tag = 'a'
        else:
            tg = pos_tag_([w])[0][1]
            if tg.startswith('NN'):
                pos_tag = 'n'
            elif tg.startswith('JJ'):
                pos_tag = 'a'
            elif tg.startswith('VB') or tg == 'MD':
                pos_tag = 'v'
            elif tg.startswith('RB'):
                pos_tag = 'r'
        if pos_tag is None:
            return primary, secondary
        synonyms = soup.find('section', class_='css-0 e1991neq0')

        for elem in synonyms.find_all('a', class_=class_a):
            primary[pos_tag].append(elem.text)
        for elem in synonyms.find_all('a', class_=class_b) +\
                    synonyms.find_all('a', class_=class_c) +\
                    synonyms.find_all('span', class_=class_d):
            secondary[pos_tag].append(elem.text)

        return primary, secondary

    primary, secondary = defaultdict(list), defaultdict(list)
    url = f"https://www.thesaurus.com/browse/{w}?s=t"

    try:
        driver = webdriver.Chrome('chromedriver',options=options)
    except:
        if is_task_4:
            res = {}
            for k in {'n', 'a', 'r', 'v'}:
                aux = primary[k] + secondary[k]
                res[k] = aux
            return res
        else:
            return primary, secondary
        
    try:
        driver.get(url)
    except Exception as e:
        print(e)
        driver.quit()
        if is_task_4:
            res = {}
            for k in {'n', 'a', 'r', 'v'}:
                aux = primary[k] + secondary[k]
                res[k] = aux
            return res
        else:
            return primary, secondary
        
    tabs = driver.find_elements_by_css_selector('a.css-wmquag.e9i53te7')


    for tab in tabs:
        try:
            p, s = get_info(driver)
            for k, v in p.items():
                primary[k] += v
            for k, v in s.items():
                secondary[k] += v
            tab.click()
            time.sleep(0.02)
        except:
            continue
    try:
        p, s = get_info(driver)
        for k, v in p.items():
            primary[k] += v
        for k, v in s.items():
            secondary[k] += v
    except Exception as e:
        print(e)
    finally:
        driver.quit()
        if is_task_4:
            res = {}
            for k in {'n', 'a', 'r', 'v'}:
                aux = primary[k] + secondary[k]
                res[k] = aux
            return res
        else:
            return primary, secondary


#############
# TASK 3 c)
#############
def get_syn_thesaurus_net(w, is_task_4=False):
    def get_pos_tag(t):
        t = t.lower()
        if t.startswith('n'):
            return 'n'
        elif t.startswith('adv'):
            return 'r'
        elif t.startswith('adj'):
            return 'a'
        elif t.startswith('ver'):
            return 'v'

    url = f"https://www.thesaurus.net/{w}"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')
    try:
        raw_text = soup.find('ul', class_='anchor-list').text
    except:
        if is_task_4:
            return defaultdict(list)
        else:
            return defaultdict(list), defaultdict(list)
    
    relevant_syn = raw_text.split('other synonyms')[0]
    parsed_syns = [elem.strip().split('(') for elem in relevant_syn.split(')')]

    primary = defaultdict(list)
    secondary = defaultdict(list)
    tag_counter = defaultdict(int)

    for elem in parsed_syns:
        if elem[0]:
            pos_tag = get_pos_tag(elem[-1])
            syns = re.split(r'[?.;,\/]', ' '.join(elem[:-1]).strip())
            tag_counter[pos_tag] += 1
            if tag_counter[pos_tag] > 25:
                for syn in syns:
                    secondary[pos_tag].append(syn.strip())
            else:
                for syn in syns:
                    primary[pos_tag].append(syn.strip())            

    if is_task_4:
        res = {}
        for k in {'n', 'a', 'r', 'v'}:
            aux = primary[k] + secondary[k]
            res[k] = set(aux)
        return res
    else:
        return primary, secondary



#############
# TASK 5 a
# COLUMN I
#############
def get_describe(w):
    url = f'https://describingwords.io/for/{w}'
    driver = webdriver.Chrome('chromedriver',options=options)
    driver.get(url)
    res = []
    try:
        driver.find_element_by_id("word-click-hint").click()
    except Exception as e:
        print(e)
        driver.quit()
        return res
    time.sleep(0.2)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    ys = soup.find('div', class_='words').find_all('span', class_='word-sub-item')    
    for y in ys[:50]:
        if y.text in res:
            continue
        if y.text:
            res.append(y.text)
    driver.quit()
    return res

#############
# TASK 5 b
# COLUMN J
#############
def get_related(w):
    url = f'https://relatedwords.org/relatedto/{w}'
    driver = webdriver.Chrome('chromedriver',options=options)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    ys = soup.find('div', class_='words').find_all('a', class_='item')    
    res = []
    for y in ys[:50]:
        if y.text in res:
            continue
        if y.text:
            res.append(y.text)
    driver.quit()
    return res

#############
# TASK 6
# COLUMN K
#############
def get_common_words(main_word, related_words, models, threshold, pos_tags_={'a', 'v', 'n', 'r'}):

    def sort_sim(pivot_word, words, model):
        try:
            res = []
            for w_ in words:
                res.append((w_, model.distance(pivot_word, w_)))
            res = sorted(res, key=lambda x: x[1])
        except Exception as e:
            print(str(e))
        return [e[0] for e in res]

    def get_intersect(sorted_results, th):
        counter = defaultdict(int)
        remainder = []
        for sorted_words in sorted_results:
            remainder += sorted_words[th:]
            for word in sorted_words[:th]:
                counter[word] += 1
        final = defaultdict(list)
        for k, v in counter.items():
            final[v].append(k)
        return final[4], set(remainder)

    oov = set()
    keep = set()
    words = set()
    model_concept = models['concept']
    model_fasttext = models['fttx']
    model_glove = models['glove']
    model_google = models['google']

    for k, v in related_words.items():
        for _, v2 in v.items():
            words = words.union(v2)

    for w in words:
        if w in model_concept.vocab and\
           w in model_fasttext.vocab and\
           w in model_glove.vocab and\
           w in model_google.vocab:
            keep.add(w)
        else:
            oov.add(w)

    th = int(len(keep)*threshold)
    
    cpt_res  = sort_sim(main_word, keep, model_concept)
    ftxt_res = sort_sim(main_word, keep, model_fasttext)
    glove_res = sort_sim(main_word, keep, model_glove)
    ggle_res = sort_sim(main_word, keep, model_google)

    final_words, remainder = get_intersect([cpt_res, ftxt_res, glove_res, ggle_res], th)

    pos_dict = {'primary': defaultdict(set),
                'remainder': defaultdict(set)}

    prim_ = related_words['primary']
    snd_ = related_words['secondary']
    for w_ in final_words:
        for tag_ in pos_tags_:
            if w_ in prim_[tag_] or w_ in snd_[tag_]:
                pos_dict['primary'][tag_].add(w_)

    for w_ in remainder:
        for tag_ in pos_tags_:
            if w_ in prim_[tag_] or w_ in snd_[tag_]:
                pos_dict['remainder'][tag_].add(w_)
        
    return pos_dict


###
# COLUMN L
###
def get_column_variants_per_pos(ws_):
    nouns_ = ws_['n']
    verbs_ = ws_['v']
    adjs_ = ws_['a']
    advs_ = ws_['r']

    res_ = defaultdict(set)
    
    for nn in nouns_:
        res_['n'].add(nn)
        aux = get_variants_and_derivatives(nn)
        for pos in {'n', 'v', 'a', 'r'}:
            res_[pos] = res_[pos].union(aux[pos])
            
    for vb in verbs_:
        res_['v'].add(vb)
        aux = get_variants_and_derivatives(vb)
        for pos in {'n', 'v', 'a', 'r'}:
            res_[pos] = res_[pos].union(aux[pos])

    for adj in adjs_:
        res_['a'].add(adj)
        aux = get_variants_and_derivatives(adj)
        for pos in {'n', 'v', 'a', 'r'}:
            res_[pos] = res_[pos].union(aux[pos])
    
    for adv in advs_:
        res_['r'].add(adv)
        aux = get_variants_and_derivatives(adv)
        for pos in {'n', 'v', 'a', 'r'}:
            res_[pos] = res_[pos].union(aux[pos])
            
    return res_

#############
# TASK 8
#############
def get_ant_thesaurus_com(w):
    url = f"https://www.thesaurus.com/browse/{w}?s=t"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'lxml')
    res = set()
    class_a = soup.find_all('section', class_='css-0 e1991neq0')
    if len(class_a) > 1:
        ants = class_a[1].find_all('span', class_='css-133coio etbu2a32')
        for a_ in ants:
            res.add(a_.text)
    return res

def get_ant_antonymfor(w):
    url = f"http://www.antonymfor.com/{w}"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "lxml")
    check_ = [x.text for x in soup.find_all('h1')]
    if f'Antonyms for {w.lower()}' not in check_:
        return set()
    else:
        mask = soup.find('div', class_='mask-holder')
        ants = mask.find_all('a', {'href': True, 'class': None})
        res = set()
        for ant in ants:
            res.add(ant.text)
        return res
