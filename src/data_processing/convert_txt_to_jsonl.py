import ujson
import warnings
warnings.filterwarnings("ignore")

def read_file():
    file_path = r"C:\Users\xia.he\Project\Hortisem_neu\data\raw\corpus_cleaned.txt"
    with open(file_path,'r',encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            try:
                yield {"text":line.strip()}
            except ValueError:
                continue

def convert_file():
    # Create a .jsonl file from the text
    file_jsonl = r"C:\Users\xia.he\Project\Hortisem_neu\data\raw\raw_text.jsonl"
    data = [ujson.dumps(text, escape_forward_slashes= False,ensure_ascii= False) for text in read_file()]
    with open(file_jsonl,'w',encoding="utf-8") as f:
        f.write('\n'.join(data))

if __name__ == "__main__":
    convert_file()
