import re

# List of contractions adapted from Robert MacIntyre's tokenizer.
CONTRACTIONS2 = [re.compile(r"(?i)\b(can)(not)\b"),
                 re.compile(r"(?i)\b(d)('ye)\b"),
                 re.compile(r"(?i)\b(gim)(me)\b"),
                 re.compile(r"(?i)\b(gon)(na)\b"),
                 re.compile(r"(?i)\b(got)(ta)\b"),
                 re.compile(r"(?i)\b(lem)(me)\b"),
                 re.compile(r"(?i)\b(mor)('n)\b"),
                 re.compile(r"(?i)\b(wan)(na) ")]
CONTRACTIONS3 = [re.compile(r"(?i) ('t)(is)\b"),
                 re.compile(r"(?i) ('t)(was)\b")]
CONTRACTIONS4 = [re.compile(r"(?i)\b(whad)(dd)(ya)\b"),
                 re.compile(r"(?i)\b(wha)(t)(cha)\b")]


def tokenize(text):
    """Regex-based tokenization. Modified from nltk TreebankWordTokenizer to handle left single quote.
    """
    # starting quotes
    text = re.sub(r'^(\"|\'\')', r'``', text)
    text = re.sub(r'(``)', r' \1 ', text)
    text = re.sub(r'([ (\[{<])("|\'\')', r'\1 `` ', text)
    text = re.sub(r"^'", r'` ', text)
    text = re.sub(r"([ (\[{<])'", r'\1 ` ', text)

    # punctuation
    text = re.sub(r'([:,])([^\d])', r' \1 \2', text)
    text = re.sub(r'\.\.\.', r' ... ', text)
    text = re.sub(r'[;@#$%&]', r' \g<0> ', text)
    text = re.sub(r'([^\.])(\.)([\]\)}>"\']*)\s*$', r'\1 \2\3 ', text)
    text = re.sub(r'[?!]', r' \g<0> ', text)

    # Tokenize left single quote
    text = re.sub(r"([^' ])('[sS]|'[mM]|'[dD]|') ", r"\1 \2 ", text)
    text = re.sub(r"([^' ])('ll|'re|'ve|n't|) ", r"\1 \2 ", text)
    text = re.sub(r"([^' ])('LL|'RE|'VE|N'T|) ", r"\1 \2 ", text)
    text = re.sub(r" '((?!['tTnNsSmMdD] |\s|[2-9]0s |em |till |cause |ll |LL |ve |VE |re |RE )\S+)", r" ` \1", text)

    # parens, brackets, etc.
    text = re.sub(r'[\]\[\(\)\{\}\<\>]', r' \g<0> ', text)
    text = re.sub(r'--', r' -- ', text)

    # add extra space to make things easier
    text = " " + text + " "

    # ending quotes
    text = re.sub(r'"', " '' ", text)
    text = re.sub(r"''", " '' ", text)
    text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)

    for regexp in CONTRACTIONS2:
        text = regexp.sub(r' \1 \2 ', text)
    for regexp in CONTRACTIONS3:
        text = regexp.sub(r' \1 \2 ', text)

    # We are not using CONTRACTIONS4 since
    # they are also commented out in the SED scripts
    # for regexp in self.CONTRACTIONS4:
    #     text = regexp.sub(r' \1 \2 \3 ', text)

    text = text.strip()

    return text.split()

if __name__ == '__main__':
    print tokenize("Who'd think that 'that' thing, which you sent (today), was my dogs' $10,000.00 \"wrapped\" teeth!")
    while True:
        sentence = raw_input('Enter a sentence: ')
        print tokenize(sentence)
