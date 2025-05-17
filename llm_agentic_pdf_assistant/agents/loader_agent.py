import fitz

def load_and_split_pdf(path, chunk_size=500):
    doc = fitz.open(path)
    text = "\n".join(page.get_text() for page in doc)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks