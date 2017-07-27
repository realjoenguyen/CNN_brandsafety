PAREN_MAP = {'(': '-LRB-', '{': '-LCB-', '[': '-LSB-', ')': '-RRB-', '}': '-RCB-', ']': '-RSB-'}
REVERSE_PAREN_MAP = dict((v, k) for k, v in PAREN_MAP.items())


def map_paren(word):
    if word in PAREN_MAP.keys():
        return PAREN_MAP[word]
    return word


def reverse_map_paren(word):
    if word in REVERSE_PAREN_MAP.keys():
        return REVERSE_PAREN_MAP[word]
    return word
