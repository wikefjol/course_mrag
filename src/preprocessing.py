import os
import pandas as pd
import PyPDF2
import pdfplumber
from typing import List, Dict, Any
from langchain_experimental.text_splitter import SemanticChunker
import re

def clean_text(text, remove_newlines=True, remove_extra_spaces=True):
    """
    Cleans text by removing unwanted characters.

    Parameters:
        text (str): The input text.
        remove_newlines (bool): Whether to remove newline characters.
        remove_extra_spaces (bool): Whether to collapse multiple spaces into one.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""
    if remove_newlines:
        text = re.sub(r"[\\n\\t\\r]+", " ", text)
    if remove_extra_spaces:
        text = re.sub(r"\\s+", " ", text)
    return text.strip()

def read_files(filepath, file_extension=".pdf"):
    """
    Reads files from the specified directory and extracts text from PDFs.

    Parameters:
        filepath (str): Path to the directory containing files.
        file_extension (str): File extension to filter by (default is ".pdf").

    Returns:
        pd.DataFrame: DataFrame containing extracted text, file names, and page numbers.
    """
    all_data = []
    for dirpath, _, filenames in os.walk(filepath):
        for filename in filenames:
            if filename.endswith(file_extension):
                file_path = os.path.join(dirpath, filename)
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            all_data.append({
                                "file_name": filename,
                                "page_number": page.page_number,
                                "text": clean_text(text),
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(all_data)