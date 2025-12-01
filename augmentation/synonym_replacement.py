import os
import re
import nltk
import random
from nltk.wsd import lesk
from nltk.corpus import wordnet, stopwords

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_PATH = './nltk_res/nltk_data'
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_PATH)

STOPWORDS = set(stopwords.words("english"))
VALID_PATTERN = re.compile(r"^[a-zA-Z]+$")

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return None

def get_synonyms_fallback(word, pos):
    """Used when Lesk fails: get synonyms directly from WordNet."""
    syns = []
    for s in wordnet.synsets(word, pos=pos):
        for l in s.lemmas():
            syn = l.name().replace("_", " ").strip()
            if syn.lower() != word.lower() and VALID_PATTERN.match(syn):
                syns.append(syn)
    return list(set(syns))


def get_valid_synonyms(word, pos, sentence):
    """Try Lesk first, fallback to normal WordNet synonyms."""
    # Try Lesk
    sense = lesk(sentence, word, pos)
    synonyms = []

    if sense:
        for lemma in sense.lemmas():
            syn = lemma.name().replace("_", " ").strip()
            if syn.lower() != word.lower() and VALID_PATTERN.match(syn):
                synonyms.append(syn)

    # If too few synonyms, fallback to normal synsets
    if len(synonyms) < 2:
        synonyms.extend(get_synonyms_fallback(word, pos))

    # Cleaning filters
    final = []
    for syn in synonyms:
        if syn.lower() not in STOPWORDS and len(syn) >= 3 and syn.isalpha():
            final.append(syn)

    return list(set(final))


def synonym_replace(text, n_per_sentence=2):
    sentences = nltk.sent_tokenize(text)
    new_sentences = []

    for sent in sentences:
        tokens = nltk.word_tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)

        candidates = []

        for i, (word, tag) in enumerate(pos_tags):
            if len(word) < 3 or word.lower() in STOPWORDS:
                continue

            wn_pos = get_wordnet_pos(tag)
            if not wn_pos:
                continue

            syns = get_valid_synonyms(word, wn_pos, tokens)
            if syns:
                candidates.append((i, word, syns))

        if not candidates:
            new_sentences.append(sent)
            continue

        random.shuffle(candidates)
        selected = candidates[:n_per_sentence]

        new_tokens = tokens[:]
        for idx, old, synlist in selected:
            new_tokens[idx] = random.choice(synlist)

        new_sentences.append(" ".join(new_tokens))

    return " ".join(new_sentences)
