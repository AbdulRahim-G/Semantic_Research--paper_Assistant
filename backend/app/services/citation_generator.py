from typing import Dict, List, Optional

class CitationGenerator:
    @staticmethod
    def generate_citations(title: str, authors_list: List[str], year: Optional[int], journal: Optional[str] = None, doi: Optional[str] = None) -> Dict[str, str]:
        """
        Generates citations in APA, MLA, and IEEE styles based on metadata.
        """
        # Fallback values
        pub_year = year if year else "n.d."
        journal_name = journal if journal and journal != "N/A" else "Research Repository"
        
        if not authors_list:
            authors_list = ["Unknown Author"]

        # Parse initials for APA and IEEE
        formatted_authors_apa = []
        formatted_authors_ieee = []
        
        for author in authors_list:
            parts = [p.strip() for p in author.split(" ") if p.strip()]
            if len(parts) > 1:
                last_name = parts[-1]
                first_initial = parts[0][0]
                formatted_authors_apa.append(f"{last_name}, {first_initial}.")
                formatted_authors_ieee.append(f"{first_initial}. {last_name}")
            else:
                formatted_authors_apa.append(author)
                formatted_authors_ieee.append(author)

        # 1. APA Format
        # Single: Last, F. M. (Year). Title. Journal.
        # Multiple: Last, F., & Last, F. (Year)...
        if len(formatted_authors_apa) == 1:
            apa_authors = formatted_authors_apa[0]
        elif len(formatted_authors_apa) == 2:
            apa_authors = f"{formatted_authors_apa[0]} & {formatted_authors_apa[1]}"
        else:
            apa_authors = ", ".join(formatted_authors_apa[:-1]) + f", & {formatted_authors_apa[-1]}"
            
        apa_doi = f" https://doi.org/{doi}" if doi else ""
        apa_citation = f"{apa_authors} ({pub_year}). {title}. *{journal_name}*." + apa_doi

        # 2. MLA Format
        # Single: Last, First. "Title." Journal, Year.
        # Multiple: Last, First, et al. "Title." Journal, Year.
        mla_authors = authors_list[0]
        if len(authors_list) > 1:
            # Re-format first author as "Last, First" for MLA
            parts = [p.strip() for p in authors_list[0].split(" ") if p.strip()]
            if len(parts) > 1:
                mla_authors = f"{parts[-1]}, {' '.join(parts[:-1])}"
            mla_authors += ", et al"
            
        mla_citation = f"{mla_authors}. \"{title}.\" *{journal_name}*, {pub_year}."

        # 3. IEEE Format
        # Example: F. M. Last et al., "Title," Journal, Year.
        ieee_authors = formatted_authors_ieee[0]
        if len(authors_list) > 1:
            ieee_authors += " et al."
            
        ieee_citation = f"{ieee_authors}, \"{title},\" *{journal_name}*, {pub_year}."

        return {
            "apa": apa_citation,
            "mla": mla_citation,
            "ieee": ieee_citation
        }
