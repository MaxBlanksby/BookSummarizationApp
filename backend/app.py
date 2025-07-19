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


def chunk_by_size(pdf_path, size_of_text_piece, overlap):
    text = open_pdf(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    text_chunks = []
    i = 0
    chunk_idx = 1
    while i < len(text):
        text_piece = text[i:i+size_of_text_piece]
        chunk_filename = f"{pdf_name}_chunk_{chunk_idx}.txt"
        text_chunks.append((chunk_filename, text_piece))
        if i + size_of_text_piece >= len(text):
            break
        i += size_of_text_piece - overlap
        chunk_idx += 1
    return text_chunks



def chunk_by_chapter(pdf_path, overlap):
    text = open_pdf(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    chapters = text.split("Chapter")
    chapter_chunks = []
    for i, chapter in enumerate(chapters):
        chapter = chapter.strip()
        if not chapter:
            continue
        if i != 0:
            chapter = "Chapter" + chapter
        chapter_chunks.append(chapter)
    overlapped_chunks = []
    for i, chunk in enumerate(chapter_chunks):
        if overlap > 0 and i > 0:
            prev_chunk = chapter_chunks[i-1]
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
            chunk = overlap_text + chunk
        chunk_filename = f"{pdf_name}_chapter_{i+1}.txt"
        overlapped_chunks.append((chunk_filename, chunk))
    return overlapped_chunks

def save_chunks_to_folder(chunks):
    os.makedirs("textFiles", exist_ok=True)
    for chunk_filename, chunk in chunks:
        pdf_name = chunk_filename.split("_chunk_")[0].split("_chapter_")[0]
        folder_path = os.path.join("textFiles", pdf_name)
        os.makedirs(folder_path, exist_ok=True)
        chunk_path = os.path.join(folder_path, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(f"{chunk_filename}\n{chunk}")
    return folder_path



def generate_test_pdf(pdf_path, size_of_text_piece):
    random_text = ''.join(random.choices(string.ascii_letters + string.digits, k=size_of_text_piece))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, random_text)
    pdf.output(pdf_path)
          
        

#generate_test_pdf("sample.pdf",10000)


chunks = chunk_by_size("Pdfs/sample.pdf", 1000, 100)
save_chunks_to_folder(chunks)









    