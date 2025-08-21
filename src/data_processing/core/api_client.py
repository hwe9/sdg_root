import requests
import json
import re
from typing import Dict, Any

class ApiClient:
    def __init__(self):
        self.headers = {'User-Agent': 'SDG-KI-Project/1.0 (info@example.com)'}

    def get_metadata_from_doi(self, doi: str) -> Dict[str, Any]:
        print(f"Abfrage der CrossRef-API für DOI: {doi}")
        api_url = f"https://api.crossref.org/works/{doi}"
        try:
            response = requests.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            message = data.get('message', {})
            metadata = {
                'title': message.get('title', [])[0] if message.get('title') else None,
                'authors': ', '.join([author.get('given', '') + ' ' + author.get('family', '') for author in message.get('author', [])]),
                'publication_year': message.get('issued', {}).get('date-parts', [[None]]),
                'publisher': message.get('publisher'),
                'doi': message.get('DOI'),
                'keywords': ", ".join(message.get('subject', [])) if 'subject' in message else None,
                'abstract_original': message.get('abstract'),
                # ggf. weitere Felder parsen und zuweisen
            }
            return metadata
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            print(f"Fehler bei der CrossRef-Abfrage für {doi}: {e}")
            return {}

    def get_metadata_from_isbn(self, isbn: str) -> Dict[str, Any]:
        print(f"Abfrage der Google Books API für ISBN: {isbn}")
        api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        try:
            response = requests.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('totalItems') == 0:
                return {}
            item = data['items'][0]['volumeInfo']
            metadata = {
                'title': item.get('title'),
                'authors': ', '.join(item.get('authors', [])),
                'publisher': item.get('publisher'),
                'publication_year': item.get('publishedDate', 'Unknown').split('-'),
                'isbn': isbn,
                'abstract_original': item.get('description'),
                # ggf. weitere Felder parsen und zuweisen
            }
            return metadata
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            print(f"Fehler bei der Google Books API-Abfrage für {isbn}: {e}")
            return {}

    def get_metadata_from_un_digital_library(self, query: str) -> Dict[str, Any]:
        print(f"Abfrage der UN Digital Library für Suchbegriff: {query}")
        api_url = f"https://digitallibrary.un.org/record?format=json&searchTerm={query}"
        try:
            response = requests.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and 'results' in data:
                first_result = data['results'][0]['value']
                metadata = {
                    'title': first_result.get('title'),
                    'authors': first_result.get('authors_names'),
                    'publication_year': first_result.get('publication_date'),
                    'source_url': first_result.get('url'),
                    # ggf. weitere Felder
                }
                return metadata
            return {}
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            print(f"Fehler bei der UN Digital Library Abfrage: {e}")
            return {}

    def get_metadata_from_oecd(self, dataset_id: str) -> Dict[str, Any]:
        print(f"Abfrage der OECD API für Dataset: {dataset_id}")
        api_url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,{dataset_id}@DF_NAAG_I?format=jsondata"
        try:
            response = requests.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            metadata = {
                'title': f"OECD Dataset: {dataset_id}",
                'publisher': "OECD",
                'source_url': api_url
            }
            return metadata
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            print(f"Fehler bei der OECD API-Abfrage: {e}")
            return {}

    def get_metadata_from_world_bank(self, query: str) -> Dict[str, Any]:
        print(f"Abfrage der Weltbank API für Suchbegriff: {query}")
        api_url = f"https://search.worldbank.org/api/v3/wds?format=json&qterm={query}"
        try:
            response = requests.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and 'documents' in data:
                first_result = data['documents'][0]
                metadata = {
                    'title': first_result.get('title'),
                    'authors': first_result.get('authors_names'),
                    'publication_year': first_result.get('pub_date'),
                    'source_url': first_result.get('url')
                }
                return metadata
            return {}
        except (requests.exceptions.RequestException, IndexError, KeyError) as e:
            print(f"Fehler bei der Weltbank API-Abfrage: {e}")
            return {}
