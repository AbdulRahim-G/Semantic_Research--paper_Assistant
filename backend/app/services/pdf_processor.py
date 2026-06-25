# pyrefly: ignore [missing-import]
import fitz  # PyMuPDF
import re
import io
import logging
from PIL import Image
from typing import Dict, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

try:
    # pyrefly: ignore [missing-import]
    import pytesseract
except ImportError:
    pytesseract = None
    logger.warning("pytesseract is not installed. OCR fallback for scanned PDFs will be disabled.")

class PDFProcessor:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans extracted text by standardizing whitespace, removing page headers/footers,
        and fixing hyphenation at line breaks.
        """
        if not text:
            return ""
        # Remove hyphenated line splits (e.g. "semi-\nconductor" -> "semiconductor")
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @classmethod
    def perform_ocr(cls, page: fitz.Page) -> str:
        """
        Renders a PDF page to an image and runs Tesseract OCR to extract text.
        """
        if not pytesseract:
            logger.error("OCR requested but pytesseract library is not available.")
            return ""
        
        try:
            # Render page to low-res image (150 DPI is usually sufficient for OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            image_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(image_data))
            
            # Run OCR
            ocr_text = pytesseract.image_to_string(image)
            return ocr_text
        except Exception as e:
            logger.error(f"Failed to perform OCR on page {page.number + 1}: {e}")
            return ""

    @classmethod
    def extract_metadata_heuristics(cls, first_page_text: str, filename: str) -> Dict:
        """
        Applies robust heuristics to extract title, authors, and year from the first page text.
        """
        metadata = {
            "title": "",
            "authors": [],
            "year": None,
            "journal": "N/A"
        }

        # Clean up lines
        raw_lines = first_page_text.split('\n')
        cleaned_lines = []
        for line in raw_lines:
            line_str = line.strip()
            if line_str:
                cleaned_lines.append(line_str)

        # 1. Identify Journal Name and Year from headers
        year_candidate = None
        for line in cleaned_lines[:10]:
            # Look for 4 digit years
            years = re.findall(r'\b(19\d{2}|20\d{2})\b', line)
            if years:
                year_candidate = int(years[0])
                if any(k in line.lower() for k in ["journal", "elsevier", "springer", "proceedings", "transactions", "letters", "open", "science"]):
                    journal_clean = re.sub(r'\b(19\d{2}|20\d{2})\b|\d+|[\(\)]', '', line).strip()
                    if len(journal_clean) > 5:
                        metadata["journal"] = re.sub(r'\s+', ' ', journal_clean).strip(" ,.-")

        if year_candidate:
            metadata["year"] = year_candidate

        # 2. Filter out non-title header lines to isolate title and authors
        header_patterns = [
            r'available\s+online', r'received', r'accepted', r'revised', 
            r'license', r'http', r'doi', r'elsevier', r'springer', r'ieee', 
            r'issn', r'volume', r'issue', r'pp\.', r'page', r'copyright',
            r'^review\s+article$', r'^research\s+article$', r'^original\s+article$',
            r'^article\s+info$', r'^abstract$', r'^introduction$'
        ]
        
        filtered_lines = []
        for line in cleaned_lines:
            if any(re.search(pat, line, re.IGNORECASE) for pat in header_patterns):
                continue
            if re.search(r'\b\d+\s*\(\d{4}\)\s*\d+\b', line) or re.search(r'ISSN\s*\d{4}-\d{4}', line, re.IGNORECASE):
                continue
            if re.match(r'^[\d\s\-\/,\.]+$', line):
                continue
            filtered_lines.append(line)

        # 3. Extract Title & Identify Authors Line
        title_lines = []
        author_line_idx = -1
        
        for idx, line in enumerate(filtered_lines[:10]):
            # An author line typically contains commas, capitalized names, and lacks institution keywords
            is_author_candidate = (
                (',' in line or ' and ' in line or '&' in line) 
                and any(word[0].isupper() for word in re.findall(r'\b\w+\b', line) if word)
                and not any(aff in line.lower() for aff in [
                    'university', 'dept', 'department', 'institute', 'school', 'labs', 
                    'college', 'sciences', 'faculty', 'saint-petersburg', 'russia', 
                    'yemen', 'oman', 'indonesia', 'yogyakarta', 'jakarta', 'bandung'
                ])
            )
            if is_author_candidate:
                author_line_idx = idx
                break

        if author_line_idx != -1:
            title_lines = filtered_lines[:author_line_idx]
            authors_raw_str = filtered_lines[author_line_idx]
        else:
            if len(filtered_lines) >= 3:
                title_lines = filtered_lines[:2]
                authors_raw_str = filtered_lines[2]
            elif filtered_lines:
                title_lines = [filtered_lines[0]]
                authors_raw_str = ""
            else:
                title_lines = [filename.replace(".pdf", "")]
                authors_raw_str = ""

        title = " ".join(title_lines).strip()
        metadata["title"] = re.sub(r'\s+', ' ', title)[:250]

        # 4. Clean and Format Author Names
        if authors_raw_str:
            raw_authors = re.split(r'[,&]|\band\b', authors_raw_str)
            cleaned_authors = []
            for raw_a in raw_authors:
                a_str = raw_a.strip()
                if not a_str:
                    continue
                # Clean trailing stars, single affiliation letters/numbers, e.g., "Al-Ansi a,*" -> "Al-Ansi"
                a_clean = re.sub(r'[\*\d\s,\x00-\x1f]+$', '', a_str)
                a_clean = re.sub(r'\s+[a-z]$', '', a_clean)
                a_clean = a_clean.strip()
                
                if len(a_clean) > 3 and any(c.isupper() for c in a_clean) and not any(aff in a_clean.lower() for aff in ['university', 'dept', 'faculty', 'institute']):
                    cleaned_authors.append(a_clean)
            
            metadata["authors"] = cleaned_authors

        if not metadata["authors"] and filename:
            metadata["title"] = filename.replace(".pdf", "")

        return metadata

    @classmethod
    def extract_references(cls, full_text: str) -> List[str]:
        """
        Attempts to locate and extract reference bibliography items from the end of the text.
        """
        ref_patterns = [
            r'\b(references|bibliography|works cited)\b'
        ]
        
        references_section = ""
        for pattern in ref_patterns:
            matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
            if matches:
                # Take the last match, which is usually the bibliography at the end
                last_match = matches[-1]
                references_section = full_text[last_match.end():]
                break
                
        if not references_section:
            return []
            
        # Parse bibliography items. They usually start with [1], [2], or (Author, Year) or Author (Year)
        # Let's split by numbered references first e.g. "[1]" or "1." at beginning of lines
        ref_items = re.split(r'\n\s*(?:\[\d+\]|\b\d+\.)\s*', references_section)
        if len(ref_items) <= 1:
            # Try splitting by typical line breaks of references (approx. 50-300 chars)
            ref_items = [ref.strip() for ref in references_section.split('\n') if len(ref.strip()) > 15]
            
        cleaned_refs = []
        for ref in ref_items:
            clean = cls.clean_text(ref)
            if len(clean) > 20: # Exclude fragments
                cleaned_refs.append(clean)
                
        return cleaned_refs

    @classmethod
    def process_pdf(cls, file_bytes: bytes, filename: str) -> Dict:
        """
        Main entry point for extracting layout-cleaned text, metadata, page maps, and references.
        """
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        raw_metadata = doc.metadata
        pages_content = []
        full_raw_text = []

        # Determine if PDF needs OCR (e.g. check first 3 pages)
        is_scanned = True
        sample_pages = min(3, len(doc))
        for i in range(sample_pages):
            page_text = doc.load_page(i).get_text()
            if len(page_text.strip()) > 150:
                is_scanned = False
                break

        logger.info(f"Processing '{filename}'. Detect scanned={is_scanned}, Total pages={len(doc)}")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = ""
            
            if not is_scanned:
                text = page.get_text("text") # Extract standard text preserving flow
            
            # OCR Fallback
            if len(text.strip()) < 50:
                logger.info(f"Page {page_num + 1} of '{filename}' has low text volume. Running OCR fallback...")
                text = cls.perform_ocr(page)
                
            full_raw_text.append(text)
            cleaned = cls.clean_text(text)
            
            pages_content.append({
                "page_number": page_num + 1,
                "text": cleaned
            })

        complete_text = "\n\n".join(full_raw_text)
        
        # Metadata Extraction
        # Merge PDF metadata dictionary with heuristics on the first page
        heuristics_meta = cls.extract_metadata_heuristics(full_raw_text[0] if full_raw_text else "", filename)
        
        final_metadata = {
            "title": raw_metadata.get("title") or heuristics_meta["title"] or filename.replace(".pdf", ""),
            "authors": heuristics_meta["authors"] or ([raw_metadata.get("author")] if raw_metadata.get("author") else []),
            "year": heuristics_meta["year"] or (int(raw_metadata.get("creationDate")[2:6]) if raw_metadata.get("creationDate") and len(raw_metadata.get("creationDate")) > 6 and raw_metadata.get("creationDate")[2:6].isdigit() else None),
            "journal": heuristics_meta["journal"] or raw_metadata.get("subject") or "N/A"
        }

        # Extract references list
        references = cls.extract_references(complete_text)

        return {
            "metadata": final_metadata,
            "pages": pages_content,
            "references": references,
            "is_scanned": is_scanned
        }
