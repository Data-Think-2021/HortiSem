import spacy
from spacy.training import Corpus

import csv


nlp = spacy.load("de_core_news_sm")
# nlp = spacy.blank("de")
corpus = Corpus(r"C:\Users\xia.he\Project\Hortisem_neu\data\from_prodigy_to_spacy\train.spacy")

data = corpus(nlp)

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
            for token, tag in zip(doc,tags):
                # clean the corpus
                if token.text == "NA":
                    print("bad text")
                else:
                    row = [n_ex, token.text, token.pos_, tag] 
                    corpus.append(row)
            corpus.append([])
    return corpus

   

def write_file(filepath):
    with open(filepath, 'w', encoding='utf-8',newline='') as f:
        writer =csv.writer(f,delimiter='\t')
        header = ['ex#','token','pos_tag','ner_tag']
        corpus = generate_corpus()
        writer.writerow(header)
        writer.writerows(corpus)
        
def main():
    write_file('./data/processed/bilou_format/train_clean.tsv')


if __name__ == '__main__':
    main()