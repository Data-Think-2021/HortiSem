import os
import sys
from pathlib import Path

import json
import re
import ujson
import warnings
warnings.filterwarnings("ignore")

from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser


pdf_dir = r'C:\Users\xia.he\Project\Hortisem_neu\data\raw\Gemuese_BB_BW_TH_SN\Test_data'
text_dir = r'C:\Users\xia.he\Project\Hortisem_neu\data\processed\input_stream_4_prodigy\gemuese_test_data.jsonl'

# PDF file Prodigy streams generators
# def get_pdf_stream(pdf_dir):
#     for root, dirs, files in os.walk(pdf_dir):
#         for pdf_file in files:
#             path_to_pdf = os.path.join(root, pdf_file)
#             [stem, ext] = os.path.splitext(path_to_pdf)
#             if ext ==".pdf":
#                 pdf_contents = read_pdf(path_to_pdf)
#                 yield {'text': pdf_contents}


def read_pdf(file_path):
    output_string = StringIO()
    with open(file_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        try: 
            doc = PDFDocument(parser)
            rsrcmgr = PDFResourceManager()
            device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
            interpreter = PDFPageInterpreter(rsrcmgr, device)
            for page in PDFPage.create_pages(doc):
                try: 
                    interpreter.process_page(page)
                except:
                    print("not able to read the pdf file")
                    pass
        except:
            print("not able to read pdf file")
            pass
        # return_string = preprocess_pdf(output_string.getvalue())
        return_string = output_string.getvalue()
        return return_string   


def preprocess_pdf(string):
    string_final = ""
    # print(string+"\n")
    bla = string.split("\n")
    for line in bla:
        string_final += line
    string_final = re.sub(r"\f[0-9]?","",string_final)
    string_final = re.sub(r"\s{2,}?"," ",string_final)

    return string_final

# convert pdf to raw text and save in a dictionary with document IDs
def get_pdf_stream(pdf_dir):
    for root, dirs, files in os.walk(pdf_dir):
        for pdf_file in files:
            path_to_pdf = os.path.join(root, pdf_file)
            [stem, ext] = os.path.splitext(path_to_pdf)
            if ext ==".pdf":
                pdf_contents = read_pdf(path_to_pdf)
                # split the doc into paragraphs
                # cleared_doc = re.sub(r'(\s{1,})','',doc) # clear text
                paragraphs = re.split(r'\.\s?\n{2,}|\.\s{2,}',pdf_contents)
                # paragraphs = re.split(r'\.\s{2,}', pdf_contents)
                # paragraph starts from 1
                par_id = 1
                # loop over each paragraph to assign a id
                for par in paragraphs:
                    # cleared_par = re.sub(r'\n','',par)  # clear text (-\n)|
                    clean_par = preprocess_pdf(par)
                    # cleared_par = re.sub(r'[W]\s','W',cleared_par)  # clear text
                    yield {'text': clean_par,'meta':{'Source':pdf_file, 'Paragraph_id':par_id}}
                    par_id +=1  #update paragraph id

def save_file():
    # Create a .jsonl file from the text
    data = [ujson.dumps(text, escape_forward_slashes= False,ensure_ascii= False) for text in get_pdf_stream(pdf_dir)]
    with open(text_dir,'w',encoding="utf-8") as f:
        f.write('\n'.join(data))

if __name__ == "__main__":
    save_file()
