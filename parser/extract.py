import pdfplumber
from docx import Document
from abc import ABC, abstractmethod

class Extractor(ABC):
    """
    Abstract base class for extracting text from files.
    """
    @abstractmethod
    def extract(self, path):
        """
        Abstract method to extract text from files.
        :param path: the file path
        :return: extracted text
        """
        pass

class PDFExtractor(Extractor):
    """
    Extract text from PDF files.
    """
    def extract(self, path):
        """
        Extract text from PDF file.
        :param path: the file path
        :return: extracted text
        """
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()

        return text


class DOCXExtractor(Extractor):
    """
    Extract text from DOCX files.
    """
    def extract(self, path):
        """
        Extract text from DOCX file.
        :param path: the file path
        :return: extracted text
        """
        text = ""
        document = Document(path)
        for paragraph in document.paragraphs:
            text += paragraph.text


class TextExtractor(Extractor):
    """
    Extract text from .txt file.
    """
    def extract(self, path):
        """
        Extract text from .txt file.
        :param path: the file path
        :return: extracted text
        """
        with open(path, "r") as file:
            text = file.read()

        return text
