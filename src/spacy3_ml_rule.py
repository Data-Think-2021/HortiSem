import spacy
from pathlib import Path

base_path = Path(r"C:\Users\xia.he\Project\Hortisem_neu")

nlp = spacy.load(base_path/"models/Spacy3/ner_4_classes/model-best")
ruler = nlp.add_pipe("entity_ruler")

# the path to the entity patterns
patterns_path = base_path/'data/raw/patterns/merged_patterns.jsonl'


new_ruler = ruler.from_disk(patterns_path)

nlp.to_disk("./models/Spacy3/ner_4_classes/ml_rule_model")

doc = nlp("BBCH 15, Rostock, Fliegen, Flugbrand . Brandenburg, in Berlin,BBCH 13-48, BBCH 3 bis 34 Schnecken, 12.1.2025")

print([(ent.text, ent.label_) for ent in doc.ents])

# for f in all_pattern_files:
#     with jsonlines.open(f) as reader:
#         ruler.add_patterns(reader)
#         nlp.add_pipe(ruler,name=f"entity_ruler_{f.stem}")