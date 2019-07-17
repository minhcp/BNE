from tqdm import tqdm
import string
import spacy

# Declare special word/character.
START_W = '<w>'
END_W = '</w>'
START_S = '<s>'
END_S = '</s>'
DICT_SW = '</dsw>'
EOL_TK = '»'
UNK = '</unk>'
PAD = '</pad>'
RESERVE_TKS = [START_W, END_W, START_S, END_S, DICT_SW, UNK, PAD, EOL_TK]

PUCT_SET = set(string.punctuation)
STOP_WORDS = set(['an', 'the', 'and', 'of', 'in', 'on', 'at',
                  'other', 'others', 'The', 'nos', 'NOS'])
STOP_TOKENS = STOP_WORDS | PUCT_SET

spacy_model = spacy.load("en_core_web_sm")


def preprocess_name(name):
    parsed_result = spacy_model(name, disable=['parser', 'tagger', 'ner'])
    tokens = [str(_) for _ in parsed_result]
    tokens = [_ for _ in tokens if _.strip() != '' and _.lower() not in STOP_TOKENS]
    new_name = ' '.join(tokens)
    return name.lower() if new_name == '' else new_name.lower()


def chunk(items, chunk_size):
    try:
        items_len = len(items)
    except:
        items_len = items.shape[0]
    for i in range(0, items_len, chunk_size):
        yield items[i:min(i + chunk_size, items_len)]


def load_pretrained_we(file_name, word_set=None):
    print('Read pretrained embeddings from {}.'.format(file_name))
    dim = None
    emb_dict = {}
    with open(file_name, encoding='utf8') as f:
        for line in tqdm(f):
            if len(line) < 20:
                continue

            k = line.find('\t')
            if k == -1:
                k = line.find(' ')
            assert k > -1

            line = line.strip()
            if word_set is None or line[:k] in word_set:
                values = line.split()
                if dim is None:
                    dim = len(values) - 1
                else:
                    assert len(values) == dim + 1
                w = values[0]
                if word_set is None or w in word_set:
                    v = [float(_) for _ in values[-dim:]]
                    emb_dict[w] = v
    return emb_dict
