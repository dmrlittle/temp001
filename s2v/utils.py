from base64 import b64encode, b64decode
from json import dumps, loads, JSONEncoder, dump, load
from itertools import chain
import pickle

def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(b64decode(dct['_python_object'].encode('utf-8')))
    return dct

def load_json_set(fpath):
    with open(fpath) as fp:
        return loads(load(fp) ,object_hook=as_python_object)

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
