import os
import openai 



openai.api_key = os.getenv("OPENAI_API_KEY")


#helper functions 
def open_pdf(pdf_path):
    return open(pdf_path, "rb")

def create_text_file(text_file_path):
    return open("output.txt", "w")

def parse_pdf(pdf_path, size_of_text_piece):
    with open_pdf(pdf_path) as pdf_file:
        text = pdf_file.read()
        text_chunks = []
        for i in range(0, len(text), size_of_text_piece):
            text_piece = text[i:i+size_of_text_piece]
            text_chunks.append(text_piece)
    return text_chunks













    