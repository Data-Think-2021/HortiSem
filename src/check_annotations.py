import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("de")
doc_bin = DocBin().from_disk("./data/from_prodigy_to_spacy/train.spacy")  # your file here
examples = []  # examples in Prodigy's format
for doc in doc_bin.get_docs(nlp.vocab):
    spans = [{"start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
    examples.append({"text": doc.text, "spans": spans})

print(examples)
# with open("examples.txt", "w", encoding="utf-8") as f:
#     f.write("\n".join(examples))