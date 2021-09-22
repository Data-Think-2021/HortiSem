import spacy
from spacy.training import Corpus

nlp = spacy.load("de_core_news_sm")
# nlp = spacy.blank("de")
corpus = Corpus(r"C:\Users\xia.he\Project\Hortisem_neu\data\from_prodigy_to_spacy\dev.spacy")

data = corpus(nlp)

def rename_biluo_to_bioes(old_tag):
    new_tag = ""
    try:
        if old_tag.startswith("L"):
            new_tag = "E" + old_tag[1:]
        elif old_tag.startswith("U"):
            new_tag = "S" + old_tag[1:]
        else:
            new_tag = old_tag
    except:
        pass
    return new_tag


def generate_corpus():
    corpus = []
    n_ex = 0
    for example in data:
        n_ex += 1
        text = example.text
        doc = nlp(text)
        tags = example.get_aligned_ner()
        # Check if it's empty list of NER tags.
        if None in tags:
            pass
        else:
            new_tags = [rename_biluo_to_bioes(tag) for tag in tags]
            for token, tag in zip(doc,new_tags):
                row = token.text +' '+ token.pos_ +' ' +tag + '\n'
                corpus.append(row)
            corpus.append('\n')
    return corpus

def write_file(filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        corpus = generate_corpus()
        # print(corpus)
        f.writelines(corpus)
        
def main():
    write_file('./data/processed/bilou_format/dev.txt')


if __name__ == '__main__':
    main()