import os
import openai 
import random
import string
from fpdf import FPDF
import PyPDF2



openai.api_key = os.getenv("OPENAI_API_KEY")


def open_pdf(pdf_path):
    pdf_file = open(pdf_path, "rb")
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    pdf_file.close()
    return text

def create_text_file(text_file_path):
    return open("output.txt", "w")

def parse_pdf(pdf_path, size_of_text_piece):
    text = open_pdf(pdf_path)
    text_chunks = []
    for i in range(0, len(text), size_of_text_piece):
        text_piece = text[i:i+size_of_text_piece]
        text_chunks.append(text_piece)
    return text_chunks


def generate_test_pdf(pdf_path, size_of_text_piece):
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=size_of_text_piece))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, random_text)
    pdf.output(pdf_path)
          
        

#generate_test_pdf("sample.pdf",10000)


os.makedirs("textFiles", exist_ok=True)
newtext = parse_pdf("Pdfs/sample.pdf", 1000)

for idx, chunk in enumerate(newtext):
    with open(f"textFiles/chunk_{idx+1}.txt", "w", encoding="utf-8") as f:
        f.write(chunk)

















    