import os
import re
import time
import json
import sqlite3
import hashlib
import secrets
import logging
import threading
from datetime import datetime, timedelta
from functools import wraps
from contextlib import contextmanager

import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.exceptions import NotFittedError

from flask import (Flask, request, jsonify, session, send_from_directory,
                   render_template_string, g)
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__, static_folder='.', static_url_path='')
app.secret_key = os.environ.get('SESSION_SECRET', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = 'medical.db'
RATE_LIMIT_STORE = {}
SCRAPE_CACHE = {}
CACHE_TTL = 3600

ML_MODELS = {
    'logistic': None,
    'linear': None,
    'kmeans': None,
    'scaler': None,
    'label_encoder': None,
    'trained': False,
    'last_trained': None,
    'disease_names': []
}

MASTER_SYMPTOM_LIST = [
    'fever', 'cough', 'headache', 'fatigue', 'nausea', 'vomiting',
    'diarrhea', 'chest_pain', 'shortness_of_breath', 'dizziness',
    'muscle_pain', 'joint_pain', 'sore_throat', 'runny_nose',
    'congestion', 'chills', 'sweating', 'rash', 'abdominal_pain',
    'back_pain', 'weight_loss', 'appetite_loss', 'insomnia',
    'anxiety', 'depression', 'blurred_vision', 'numbness',
    'tingling', 'swelling', 'bruising', 'bleeding', 'itching',
    'dry_mouth', 'frequent_urination', 'blood_in_urine',
    'constipation', 'heartburn', 'difficulty_swallowing',
    'ear_pain', 'eye_pain', 'neck_pain', 'palpitations',
    'weakness', 'confusion', 'seizures', 'tremors',
    'sneezing', 'wheezing', 'night_sweats', 'hair_loss'
]

EMERGENCY_SYMPTOMS = [
    'chest_pain', 'shortness_of_breath', 'seizures', 'confusion',
    'bleeding', 'difficulty_swallowing', 'palpitations', 'numbness'
]


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symptoms TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                conditions TEXT,
                vitals TEXT,
                consultation_type TEXT,
                risk_level TEXT,
                probability_score REAL,
                severity_index REAL,
                cluster_id INTEGER,
                confidence_score REAL,
                emergency_flag INTEGER DEFAULT 0,
                advisory TEXT,
                predicted_disease TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS scraped_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_name TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                severity TEXT,
                recommended_actions TEXT,
                risk_indicators TEXT,
                source_url TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                endpoint TEXT,
                method TEXT,
                status_code INTEGER,
                duration_ms REAL,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_consultations_user ON consultations(user_id);
            CREATE INDEX IF NOT EXISTS idx_scraped_disease ON scraped_data(disease_name);
        ''')


def log_request_decorator(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        start = time.time()
        try:
            response = f(*args, **kwargs)
            status = response[1] if isinstance(response, tuple) else 200
        except Exception as e:
            status = 500
            raise e
        finally:
            duration = (time.time() - start) * 1000
            try:
                with get_db() as conn:
                    conn.execute(
                        'INSERT INTO request_logs (user_id, endpoint, method, status_code, duration_ms, ip_address) VALUES (?,?,?,?,?,?)',
                        (session.get('user_id'), request.path, request.method, status, duration, request.remote_addr)
                    )
            except Exception:
                pass
        return response
    return decorated


def rate_limit(max_requests=30, window=60):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            key = f"{request.remote_addr}:{request.path}"
            now = time.time()
            if key not in RATE_LIMIT_STORE:
                RATE_LIMIT_STORE[key] = []
            RATE_LIMIT_STORE[key] = [t for t in RATE_LIMIT_STORE[key] if now - t < window]
            if len(RATE_LIMIT_STORE[key]) >= max_requests:
                return jsonify({'error': 'Rate limit exceeded. Please wait before trying again.'}), 429
            RATE_LIMIT_STORE[key].append(now)
            return f(*args, **kwargs)
        return decorated
    return decorator


def validate_input(schema):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            data = request.get_json(silent=True)
            if not data:
                return jsonify({'error': 'Invalid request body'}), 400
            for field, rules in schema.items():
                if rules.get('required') and field not in data:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
                if field in data:
                    val = data[field]
                    expected = rules.get('type')
                    if expected == 'str' and not isinstance(val, str):
                        return jsonify({'error': f'{field} must be a string'}), 400
                    if expected == 'int' and not isinstance(val, (int, float)):
                        return jsonify({'error': f'{field} must be a number'}), 400
                    if expected == 'list' and not isinstance(val, list):
                        return jsonify({'error': f'{field} must be a list'}), 400
                    if 'max_length' in rules and isinstance(val, str) and len(val) > rules['max_length']:
                        return jsonify({'error': f'{field} exceeds max length'}), 400
            g.validated_data = data
            return f(*args, **kwargs)
        return decorated
    return decorator


def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


def sanitize(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[<>"\';&]', '', text)
    return text.strip()


def cache_scrape(ttl=CACHE_TTL):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            key = f"{f.__name__}:{str(args)}:{str(kwargs)}"
            now = time.time()
            if key in SCRAPE_CACHE and now - SCRAPE_CACHE[key]['time'] < ttl:
                return SCRAPE_CACHE[key]['data']
            result = f(*args, **kwargs)
            SCRAPE_CACHE[key] = {'data': result, 'time': now}
            return result
        return decorated
    return decorator


SCRAPE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5'
}


def extract_symptoms_from_text(text):
    text_lower = text.lower()
    found = []
    for symptom in MASTER_SYMPTOM_LIST:
        readable = symptom.replace('_', ' ')
        if readable in text_lower or symptom in text_lower:
            found.append(symptom)
    pain_patterns = {
        'headache': 'headache', 'head pain': 'headache',
        'stomach pain': 'abdominal_pain', 'belly pain': 'abdominal_pain', 'tummy pain': 'abdominal_pain',
        'breathless': 'shortness_of_breath', 'difficulty breathing': 'shortness_of_breath', 'breath': 'shortness_of_breath',
        'tired': 'fatigue', 'tiredness': 'fatigue', 'exhaustion': 'fatigue', 'lack of energy': 'fatigue',
        'feeling sick': 'nausea', 'vomit': 'vomiting', 'being sick': 'vomiting',
        'high temperature': 'fever', 'temperature': 'fever', 'shivering': 'chills',
        'runny nose': 'runny_nose', 'blocked nose': 'congestion', 'stuffy nose': 'congestion',
        'skin rash': 'rash', 'spots': 'rash',
        'ache': 'muscle_pain', 'body ache': 'muscle_pain',
        'dizzy': 'dizziness', 'light-headed': 'dizziness', 'lightheaded': 'dizziness',
        'swollen': 'swelling', 'puffy': 'swelling',
        'weight gain': 'swelling', 'losing weight': 'weight_loss',
        'not hungry': 'appetite_loss', 'loss of appetite': 'appetite_loss',
        'trouble sleeping': 'insomnia', 'sleep problems': 'insomnia',
        'pins and needles': 'tingling', 'tingle': 'tingling',
        'blurry vision': 'blurred_vision', 'vision problems': 'blurred_vision',
        'racing heart': 'palpitations', 'heart racing': 'palpitations', 'irregular heartbeat': 'palpitations',
        'fits': 'seizures', 'convulsions': 'seizures',
        'shaking': 'tremors', 'trembling': 'tremors',
        'loose stools': 'diarrhea', 'watery stools': 'diarrhea',
        'passing urine': 'frequent_urination', 'urinating more': 'frequent_urination',
        'blood in pee': 'blood_in_urine', 'blood when urinating': 'blood_in_urine',
        'losing hair': 'hair_loss', 'hair thinning': 'hair_loss',
        'night sweat': 'night_sweats', 'sweating at night': 'night_sweats',
        'difficulty eating': 'difficulty_swallowing', 'hard to swallow': 'difficulty_swallowing',
        'acid reflux': 'heartburn', 'indigestion': 'heartburn',
        'irritable': 'anxiety', 'nervous': 'anxiety', 'worry': 'anxiety',
        'feeling down': 'depression', 'low mood': 'depression', 'sadness': 'depression',
    }
    for pattern, symptom in pain_patterns.items():
        if pattern in text_lower and symptom not in found:
            found.append(symptom)
    return found


def deep_scrape_disease_page(url):
    try:
        resp = requests.get(url, headers=SCRAPE_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        symptom_sections = []
        for heading in soup.find_all(['h2', 'h3', 'h4']):
            heading_text = heading.get_text(strip=True).lower()
            if any(kw in heading_text for kw in ['symptom', 'sign', 'what are the symptoms', 'when to see']):
                sibling = heading.find_next_sibling()
                section_text = ''
                while sibling and sibling.name not in ['h2', 'h3', 'h4']:
                    section_text += ' ' + sibling.get_text(' ', strip=True)
                    sibling = sibling.find_next_sibling()
                symptom_sections.append(section_text)
        if not symptom_sections:
            lists = soup.select('ul li, ol li')
            body_text = ' '.join(li.get_text(' ', strip=True) for li in lists[:50])
            symptom_sections.append(body_text)
        full_text = ' '.join(symptom_sections)
        return extract_symptoms_from_text(full_text)
    except Exception:
        return []


@cache_scrape(ttl=3600)
def scrape_medical_data():
    all_data = []
    scraped_from_pages = 0
    fallback_used = 0

    nhs_data, nhs_links = scrape_nhs_index()
    all_data.extend(nhs_data)

    mayo_data = scrape_mayo_index()
    all_data.extend(mayo_data)

    medline_data = scrape_medlineplus()
    all_data.extend(medline_data)

    for item in all_data:
        if item.get('source_type') == 'deep_scraped':
            scraped_from_pages += 1
        else:
            fallback_used += 1

    if len(all_data) < 20:
        all_data.extend(scrape_who_data())

    for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
        existing = [d['disease_name'].lower() for d in all_data]
        if disease.lower() not in existing:
            all_data.append({
                'disease_name': disease.title(),
                'symptoms': symptoms,
                'severity': compute_base_severity(symptoms),
                'recommended_actions': generate_actions(disease, symptoms),
                'risk_indicators': identify_risk_indicators(symptoms),
                'source_url': 'curated_medical_knowledge_base',
                'source_type': 'knowledge_base'
            })

    if all_data:
        store_scraped_data(all_data)

    logger.info(f"Total records: {len(all_data)} | Deep-scraped: {scraped_from_pages} | Fallback: {fallback_used} | Knowledge base: {len(all_data) - scraped_from_pages - fallback_used}")
    return all_data


def scrape_nhs_index():
    results = []
    disease_links = []
    try:
        resp = requests.get('https://www.nhsinform.scot/illnesses-and-conditions/a-to-z', headers=SCRAPE_HEADERS, timeout=15)
        if resp.status_code != 200:
            return results, disease_links
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = soup.select('a[href*="/illnesses-and-conditions/"]')
        diseases_found = set()
        for link in links[:100]:
            name = link.get_text(strip=True)
            href = link.get('href', '')
            if name and len(name) > 2 and name not in diseases_found:
                diseases_found.add(name)
                full_url = href if href.startswith('http') else f"https://www.nhsinform.scot{href}"
                disease_links.append((name, full_url))
        for name, page_url in disease_links[:25]:
            scraped_symptoms = deep_scrape_disease_page(page_url)
            if len(scraped_symptoms) >= 2:
                results.append({
                    'disease_name': name,
                    'symptoms': scraped_symptoms,
                    'severity': compute_base_severity(scraped_symptoms),
                    'recommended_actions': generate_actions(name, scraped_symptoms),
                    'risk_indicators': identify_risk_indicators(scraped_symptoms),
                    'source_url': page_url,
                    'source_type': 'deep_scraped'
                })
            else:
                fallback = extract_symptom_mapping(name)
                if fallback:
                    results.append({
                        'disease_name': name,
                        'symptoms': fallback,
                        'severity': compute_base_severity(fallback),
                        'recommended_actions': generate_actions(name, fallback),
                        'risk_indicators': identify_risk_indicators(fallback),
                        'source_url': page_url,
                        'source_type': 'fallback'
                    })
            time.sleep(0.3)
    except Exception as e:
        logger.warning(f"NHS scrape failed: {e}")
    return results, disease_links


def scrape_mayo_index():
    results = []
    try:
        for letter in ['A', 'B', 'C', 'D']:
            resp = requests.get(f'https://www.mayoclinic.org/diseases-conditions/index?letter={letter}', headers=SCRAPE_HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.select('a[href*="/diseases-conditions/"]')
            diseases_found = set()
            for link in links[:20]:
                name = link.get_text(strip=True)
                href = link.get('href', '')
                if name and len(name) > 2 and 'index' not in name.lower() and name not in diseases_found:
                    diseases_found.add(name)
                    if '/symptoms-causes' in href or '/syc-' in href:
                        full_url = href if href.startswith('http') else f"https://www.mayoclinic.org{href}"
                        scraped_symptoms = deep_scrape_disease_page(full_url)
                        if len(scraped_symptoms) >= 2:
                            results.append({
                                'disease_name': name,
                                'symptoms': scraped_symptoms,
                                'severity': compute_base_severity(scraped_symptoms),
                                'recommended_actions': generate_actions(name, scraped_symptoms),
                                'risk_indicators': identify_risk_indicators(scraped_symptoms),
                                'source_url': full_url,
                                'source_type': 'deep_scraped'
                            })
                            continue
                    fallback = extract_symptom_mapping(name)
                    if fallback:
                        results.append({
                            'disease_name': name,
                            'symptoms': fallback,
                            'severity': compute_base_severity(fallback),
                            'recommended_actions': generate_actions(name, fallback),
                            'risk_indicators': identify_risk_indicators(fallback),
                            'source_url': f'https://www.mayoclinic.org/diseases-conditions/index?letter={letter}',
                            'source_type': 'fallback'
                        })
                time.sleep(0.2)
    except Exception as e:
        logger.warning(f"Mayo Clinic scrape failed: {e}")
    return results


def scrape_medlineplus():
    results = []
    try:
        resp = requests.get('https://medlineplus.gov/healthtopics.html', headers=SCRAPE_HEADERS, timeout=15)
        if resp.status_code != 200:
            return results
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = soup.select('a[href*="medlineplus.gov/"]')
        diseases_found = set()
        for link in links[:60]:
            name = link.get_text(strip=True)
            href = link.get('href', '')
            if name and len(name) > 3 and len(name) < 60 and name not in diseases_found:
                diseases_found.add(name)
                if href.startswith('http') and 'medlineplus.gov' in href and '.html' in href:
                    scraped_symptoms = deep_scrape_disease_page(href)
                    if len(scraped_symptoms) >= 2:
                        results.append({
                            'disease_name': name,
                            'symptoms': scraped_symptoms,
                            'severity': compute_base_severity(scraped_symptoms),
                            'recommended_actions': generate_actions(name, scraped_symptoms),
                            'risk_indicators': identify_risk_indicators(scraped_symptoms),
                            'source_url': href,
                            'source_type': 'deep_scraped'
                        })
                        continue
                fallback = extract_symptom_mapping(name)
                if fallback:
                    results.append({
                        'disease_name': name,
                        'symptoms': fallback,
                        'severity': compute_base_severity(fallback),
                        'recommended_actions': generate_actions(name, fallback),
                        'risk_indicators': identify_risk_indicators(fallback),
                        'source_url': 'https://medlineplus.gov/healthtopics.html',
                        'source_type': 'fallback'
                    })
            time.sleep(0.2)
    except Exception as e:
        logger.warning(f"MedlinePlus scrape failed: {e}")
    return results


def scrape_who_data():
    results = []
    try:
        resp = requests.get('https://www.who.int/health-topics', headers=SCRAPE_HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            links = soup.select('a[href*="/health-topics/"]')
            seen = set()
            for link in links[:40]:
                name = link.get_text(strip=True)
                href = link.get('href', '')
                if name and len(name) > 2 and name not in seen:
                    seen.add(name)
                    full_url = href if href.startswith('http') else f"https://www.who.int{href}"
                    scraped_symptoms = deep_scrape_disease_page(full_url)
                    if len(scraped_symptoms) >= 2:
                        results.append({
                            'disease_name': name,
                            'symptoms': scraped_symptoms,
                            'severity': compute_base_severity(scraped_symptoms),
                            'recommended_actions': generate_actions(name, scraped_symptoms),
                            'risk_indicators': identify_risk_indicators(scraped_symptoms),
                            'source_url': full_url,
                            'source_type': 'deep_scraped'
                        })
                    else:
                        fallback = extract_symptom_mapping(name)
                        if fallback:
                            results.append({
                                'disease_name': name,
                                'symptoms': fallback,
                                'severity': compute_base_severity(fallback),
                                'recommended_actions': generate_actions(name, fallback),
                                'risk_indicators': identify_risk_indicators(fallback),
                                'source_url': full_url,
                                'source_type': 'fallback'
                            })
                    time.sleep(0.3)
    except Exception as e:
        logger.warning(f"WHO scrape failed: {e}")
    return results


DISEASE_SYMPTOM_MAP = {
    'flu': ['fever', 'cough', 'fatigue', 'muscle_pain', 'headache', 'chills', 'sore_throat', 'congestion', 'sweating'],
    'influenza': ['fever', 'cough', 'fatigue', 'muscle_pain', 'headache', 'chills', 'sore_throat', 'sweating'],
    'cold': ['cough', 'runny_nose', 'congestion', 'sore_throat', 'sneezing', 'fatigue', 'headache'],
    'common cold': ['cough', 'runny_nose', 'congestion', 'sore_throat', 'sneezing', 'fatigue'],
    'pneumonia': ['fever', 'cough', 'shortness_of_breath', 'chest_pain', 'fatigue', 'chills', 'sweating', 'nausea'],
    'bronchitis': ['cough', 'fatigue', 'shortness_of_breath', 'chest_pain', 'congestion', 'sore_throat', 'wheezing'],
    'asthma': ['shortness_of_breath', 'wheezing', 'cough', 'chest_pain', 'fatigue'],
    'covid': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'headache', 'muscle_pain', 'sore_throat', 'appetite_loss', 'diarrhea'],
    'coronavirus': ['fever', 'cough', 'fatigue', 'shortness_of_breath', 'headache', 'muscle_pain', 'appetite_loss'],
    'diabetes': ['frequent_urination', 'fatigue', 'blurred_vision', 'weight_loss', 'numbness', 'tingling', 'dry_mouth'],
    'hypertension': ['headache', 'dizziness', 'blurred_vision', 'chest_pain', 'shortness_of_breath', 'nausea', 'palpitations'],
    'heart disease': ['chest_pain', 'shortness_of_breath', 'palpitations', 'dizziness', 'fatigue', 'swelling', 'weakness'],
    'migraine': ['headache', 'nausea', 'vomiting', 'blurred_vision', 'dizziness', 'fatigue', 'numbness'],
    'gastritis': ['abdominal_pain', 'nausea', 'vomiting', 'heartburn', 'appetite_loss', 'bloating'],
    'arthritis': ['joint_pain', 'swelling', 'muscle_pain', 'fatigue', 'weakness', 'numbness', 'back_pain'],
    'anemia': ['fatigue', 'weakness', 'dizziness', 'shortness_of_breath', 'headache', 'chest_pain', 'palpitations'],
    'tuberculosis': ['cough', 'fever', 'night_sweats', 'weight_loss', 'fatigue', 'chest_pain', 'appetite_loss', 'bleeding'],
    'malaria': ['fever', 'chills', 'headache', 'sweating', 'nausea', 'vomiting', 'muscle_pain', 'fatigue'],
    'dengue': ['fever', 'headache', 'muscle_pain', 'joint_pain', 'rash', 'nausea', 'fatigue', 'bleeding'],
    'typhoid': ['fever', 'headache', 'abdominal_pain', 'fatigue', 'constipation', 'diarrhea', 'appetite_loss', 'rash'],
    'allergy': ['sneezing', 'runny_nose', 'itching', 'rash', 'swelling', 'congestion', 'eye_pain', 'wheezing'],
    'sinusitis': ['congestion', 'headache', 'fatigue', 'runny_nose', 'ear_pain', 'sore_throat', 'cough'],
    'urinary tract infection': ['frequent_urination', 'abdominal_pain', 'blood_in_urine', 'fever', 'back_pain', 'nausea'],
    'kidney disease': ['fatigue', 'swelling', 'frequent_urination', 'nausea', 'back_pain', 'blood_in_urine', 'appetite_loss', 'confusion'],
    'liver disease': ['fatigue', 'nausea', 'abdominal_pain', 'swelling', 'itching', 'weight_loss', 'bruising', 'confusion'],
    'depression': ['depression', 'fatigue', 'insomnia', 'appetite_loss', 'weight_loss', 'anxiety', 'confusion', 'headache'],
    'anxiety disorder': ['anxiety', 'palpitations', 'sweating', 'tremors', 'insomnia', 'dizziness', 'nausea', 'chest_pain'],
    'stroke': ['numbness', 'confusion', 'headache', 'dizziness', 'blurred_vision', 'weakness', 'difficulty_swallowing', 'seizures'],
    'epilepsy': ['seizures', 'confusion', 'dizziness', 'numbness', 'anxiety', 'headache', 'fatigue'],
    'meningitis': ['fever', 'headache', 'neck_pain', 'nausea', 'vomiting', 'confusion', 'seizures', 'rash', 'sensitivity'],
    'food poisoning': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'fever', 'fatigue', 'chills', 'weakness'],
    'gastroenteritis': ['diarrhea', 'nausea', 'vomiting', 'abdominal_pain', 'fever', 'fatigue', 'muscle_pain'],
    'eczema': ['itching', 'rash', 'dry_mouth', 'swelling', 'insomnia'],
    'psoriasis': ['rash', 'itching', 'joint_pain', 'fatigue', 'swelling'],
    'thyroid': ['fatigue', 'weight_loss', 'anxiety', 'tremors', 'sweating', 'palpitations', 'insomnia', 'hair_loss'],
    'cancer': ['fatigue', 'weight_loss', 'appetite_loss', 'night_sweats', 'fever', 'bleeding', 'weakness', 'back_pain'],
    'copd': ['shortness_of_breath', 'cough', 'wheezing', 'chest_pain', 'fatigue', 'weight_loss', 'swelling'],
    'gout': ['joint_pain', 'swelling', 'rash', 'fever', 'fatigue'],
    'fibromyalgia': ['muscle_pain', 'fatigue', 'insomnia', 'headache', 'depression', 'anxiety', 'numbness', 'tingling'],
    'vertigo': ['dizziness', 'nausea', 'vomiting', 'headache', 'sweating', 'ear_pain', 'blurred_vision'],
    'chickenpox': ['rash', 'fever', 'fatigue', 'headache', 'itching', 'appetite_loss'],
    'measles': ['fever', 'cough', 'rash', 'runny_nose', 'eye_pain', 'fatigue', 'sore_throat'],
    'mumps': ['fever', 'headache', 'muscle_pain', 'fatigue', 'appetite_loss', 'swelling', 'ear_pain'],
    'hepatitis': ['fatigue', 'nausea', 'abdominal_pain', 'appetite_loss', 'fever', 'joint_pain', 'itching'],
    'pancreatitis': ['abdominal_pain', 'nausea', 'vomiting', 'fever', 'back_pain', 'weight_loss', 'appetite_loss'],
    'appendicitis': ['abdominal_pain', 'nausea', 'vomiting', 'fever', 'appetite_loss', 'constipation'],
    'sciatica': ['back_pain', 'numbness', 'tingling', 'weakness', 'muscle_pain'],
    'carpal tunnel': ['numbness', 'tingling', 'weakness', 'joint_pain', 'muscle_pain'],
    'osteoporosis': ['back_pain', 'weakness', 'joint_pain', 'fatigue'],
    'ibs': ['abdominal_pain', 'diarrhea', 'constipation', 'nausea', 'fatigue', 'appetite_loss'],
}


def extract_symptom_mapping(disease_name):
    name_lower = disease_name.lower().strip()
    for key, symptoms in DISEASE_SYMPTOM_MAP.items():
        if key in name_lower or name_lower in key:
            return symptoms
    words = name_lower.split()
    for word in words:
        if len(word) > 3:
            for key, symptoms in DISEASE_SYMPTOM_MAP.items():
                if word in key or key in word:
                    return symptoms
    return None


def compute_base_severity(symptoms):
    emergency_count = sum(1 for s in symptoms if s in EMERGENCY_SYMPTOMS)
    total = len(symptoms)
    if emergency_count >= 3:
        return 'critical'
    elif emergency_count >= 2:
        return 'high'
    elif total >= 6:
        return 'medium'
    return 'low'


def generate_actions(disease, symptoms):
    actions = []
    severity = compute_base_severity(symptoms)
    if severity == 'critical':
        actions.append('Seek emergency medical attention immediately')
        actions.append('Call emergency services if symptoms worsen')
    elif severity == 'high':
        actions.append('Schedule an urgent doctor appointment')
        actions.append('Monitor symptoms closely every few hours')
    else:
        actions.append('Rest and maintain hydration')
        actions.append('Schedule a routine medical consultation')

    if 'fever' in symptoms:
        actions.append('Monitor temperature regularly')
    if 'cough' in symptoms:
        actions.append('Stay hydrated and consider warm fluids')
    if 'chest_pain' in symptoms:
        actions.append('Avoid physical exertion until evaluated')
    return actions


def identify_risk_indicators(symptoms):
    indicators = []
    if 'chest_pain' in symptoms:
        indicators.append('Cardiac risk')
    if 'shortness_of_breath' in symptoms:
        indicators.append('Respiratory risk')
    if 'seizures' in symptoms:
        indicators.append('Neurological risk')
    if 'bleeding' in symptoms:
        indicators.append('Hemorrhagic risk')
    if 'confusion' in symptoms:
        indicators.append('Cognitive risk')
    if 'numbness' in symptoms:
        indicators.append('Neuropathic risk')
    if not indicators:
        indicators.append('Standard monitoring recommended')
    return indicators


def store_scraped_data(data_list):
    with get_db() as conn:
        for item in data_list:
            existing = conn.execute(
                'SELECT id FROM scraped_data WHERE disease_name = ?',
                (item['disease_name'],)
            ).fetchone()
            if not existing:
                conn.execute(
                    'INSERT INTO scraped_data (disease_name, symptoms, severity, recommended_actions, risk_indicators, source_url) VALUES (?,?,?,?,?,?)',
                    (
                        item['disease_name'],
                        json.dumps(item['symptoms']),
                        item['severity'],
                        json.dumps(item['recommended_actions']),
                        json.dumps(item['risk_indicators']),
                        item.get('source_url', '')
                    )
                )


def symptoms_to_vector(symptom_list):
    vector = [0] * len(MASTER_SYMPTOM_LIST)
    for symptom in symptom_list:
        s = symptom.lower().strip().replace(' ', '_')
        if s in MASTER_SYMPTOM_LIST:
            vector[MASTER_SYMPTOM_LIST.index(s)] = 1
    return vector


def build_training_data():
    X = []
    y_disease = []
    y_severity = []

    with get_db() as conn:
        rows = conn.execute('SELECT disease_name, symptoms, severity FROM scraped_data').fetchall()

    if not rows:
        scraped = scrape_medical_data()
        if scraped:
            with get_db() as conn:
                rows = conn.execute('SELECT disease_name, symptoms, severity FROM scraped_data').fetchall()

    if not rows:
        for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
            vec = symptoms_to_vector(symptoms)
            X.append(vec)
            y_disease.append(disease)
            sev = compute_base_severity(symptoms)
            sev_map = {'low': 0.2, 'medium': 0.5, 'high': 0.75, 'critical': 0.95}
            y_severity.append(sev_map.get(sev, 0.3))
        return np.array(X), y_disease, np.array(y_severity)

    for row in rows:
        try:
            symptoms = json.loads(row['symptoms'])
            vec = symptoms_to_vector(symptoms)
            X.append(vec)
            y_disease.append(row['disease_name'])
            sev_map = {'low': 0.2, 'medium': 0.5, 'high': 0.75, 'critical': 0.95}
            y_severity.append(sev_map.get(row['severity'], 0.3))
        except Exception:
            continue

    for disease, symptoms in DISEASE_SYMPTOM_MAP.items():
        if disease not in y_disease:
            vec = symptoms_to_vector(symptoms)
            X.append(vec)
            y_disease.append(disease)
            sev = compute_base_severity(symptoms)
            sev_map = {'low': 0.2, 'medium': 0.5, 'high': 0.75, 'critical': 0.95}
            y_severity.append(sev_map.get(sev, 0.3))

    return np.array(X), y_disease, np.array(y_severity)


def train_models():
    try:
        X, y_disease, y_severity = build_training_data()
        if len(X) < 5:
            logger.warning("Insufficient training data")
            return False

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_disease)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        log_model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
        log_model.fit(X_scaled, y_encoded)

        lin_model = LinearRegression()
        lin_model.fit(X_scaled, y_severity)

        n_clusters = min(8, max(2, len(X) // 5))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        ML_MODELS['logistic'] = log_model
        ML_MODELS['linear'] = lin_model
        ML_MODELS['kmeans'] = kmeans
        ML_MODELS['scaler'] = scaler
        ML_MODELS['label_encoder'] = le
        ML_MODELS['trained'] = True
        ML_MODELS['last_trained'] = datetime.now()
        ML_MODELS['disease_names'] = list(le.classes_)

        logger.info(f"Models trained successfully on {len(X)} samples, {len(le.classes_)} diseases, {n_clusters} clusters")
        return True
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False


def predict(symptom_vector, age=30, gender='unknown', conditions=None):
    if not ML_MODELS['trained']:
        train_models()
    if not ML_MODELS['trained']:
        return None

    X = np.array(symptom_vector).reshape(1, -1)
    X_scaled = ML_MODELS['scaler'].transform(X)

    proba = ML_MODELS['logistic'].predict_proba(X_scaled)[0]
    top_indices = np.argsort(proba)[::-1][:5]
    top_diseases = []
    for idx in top_indices:
        if proba[idx] > 0.01:
            top_diseases.append({
                'disease': ML_MODELS['disease_names'][idx],
                'probability': round(float(proba[idx]) * 100, 2)
            })

    severity_raw = ML_MODELS['linear'].predict(X_scaled)[0]
    severity_score = max(0.0, min(1.0, float(severity_raw)))

    cluster_id = int(ML_MODELS['kmeans'].predict(X_scaled)[0])
    cluster_center = ML_MODELS['kmeans'].cluster_centers_[cluster_id]
    cluster_distance = float(np.linalg.norm(X_scaled[0] - cluster_center))

    age_factor = 1.0
    if age and isinstance(age, (int, float)):
        if age > 65 or age < 5:
            age_factor = 1.3
        elif age > 50 or age < 12:
            age_factor = 1.15

    condition_factor = 1.0
    if conditions:
        high_risk_conditions = ['diabetes', 'heart disease', 'hypertension', 'cancer', 'asthma', 'copd', 'kidney', 'liver', 'hiv']
        for cond in conditions:
            if any(hrc in cond.lower() for hrc in high_risk_conditions):
                condition_factor += 0.15

    active_symptoms = sum(symptom_vector)
    emergency_active = sum(1 for i, v in enumerate(symptom_vector) if v == 1 and MASTER_SYMPTOM_LIST[i] in EMERGENCY_SYMPTOMS)

    adjusted_severity = min(1.0, severity_score * age_factor * condition_factor)
    if emergency_active >= 2:
        adjusted_severity = min(1.0, adjusted_severity + 0.25)

    cluster_density_factor = 1.0
    if cluster_distance < 1.0:
        cluster_density_factor = 1.15
    elif cluster_distance > 3.0:
        cluster_density_factor = 0.9

    final_severity = min(1.0, adjusted_severity * cluster_density_factor)

    if final_severity >= 0.8:
        risk_level = 'Critical'
    elif final_severity >= 0.6:
        risk_level = 'High'
    elif final_severity >= 0.35:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    emergency_flag = emergency_active >= 2 or final_severity >= 0.8

    max_proba = top_diseases[0]['probability'] if top_diseases else 0
    symptom_coverage = active_symptoms / max(1, len([s for s in MASTER_SYMPTOM_LIST]))
    confidence = min(95, max(15, max_proba * 0.6 + symptom_coverage * 100 * 0.2 + (1 - cluster_distance / 5) * 100 * 0.2))

    if risk_level == 'Critical' or emergency_flag:
        advisory_category = 'Emergency'
        advisory_text = 'Immediate medical attention is strongly recommended. Please contact emergency services or visit the nearest emergency room.'
    elif risk_level == 'High':
        advisory_category = 'Urgent Care'
        advisory_text = 'Prompt medical evaluation is recommended. Schedule an urgent appointment with a healthcare provider within 24 hours.'
    elif risk_level == 'Medium':
        advisory_category = 'Schedule Doctor'
        advisory_text = 'A medical consultation is advisable. Schedule an appointment with your primary care physician within the week.'
    else:
        advisory_category = 'Self-Care'
        advisory_text = 'Symptoms suggest a manageable condition. Rest, stay hydrated, and monitor your symptoms. Seek medical advice if they persist or worsen.'

    return {
        'top_diseases': top_diseases,
        'severity_index': round(final_severity, 4),
        'risk_level': risk_level,
        'cluster_id': cluster_id,
        'cluster_distance': round(cluster_distance, 4),
        'confidence_score': round(confidence, 2),
        'emergency_flag': emergency_flag,
        'advisory_category': advisory_category,
        'advisory_text': advisory_text,
        'age_factor': age_factor,
        'condition_factor': condition_factor,
        'emergency_symptoms_detected': emergency_active,
        'total_symptoms_analyzed': active_symptoms
    }


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/styles.css')
def styles():
    return send_from_directory('.', 'styles.css')


@app.route('/script.js')
def script():
    return send_from_directory('.', 'script.js')


@app.route('/api/register', methods=['POST'])
@log_request_decorator
@rate_limit(max_requests=10, window=60)
@validate_input({
    'username': {'required': True, 'type': 'str', 'max_length': 50},
    'email': {'required': True, 'type': 'str', 'max_length': 100},
    'password': {'required': True, 'type': 'str', 'max_length': 128}
})
def register():
    data = g.validated_data
    username = sanitize(data['username'])
    email = sanitize(data['email'])
    password = data['password']

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return jsonify({'error': 'Invalid email format'}), 400

    pw_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

    try:
        with get_db() as conn:
            conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?,?,?)',
                         (username, email, pw_hash))
            user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            session['user_id'] = user['id']
            session['username'] = username
            session.permanent = True
            return jsonify({'message': 'Registration successful', 'username': username}), 201
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username or email already exists'}), 409


@app.route('/api/login', methods=['POST'])
@log_request_decorator
@rate_limit(max_requests=15, window=60)
@validate_input({
    'username': {'required': True, 'type': 'str'},
    'password': {'required': True, 'type': 'str'}
})
def login():
    data = g.validated_data
    username = sanitize(data['username'])
    with get_db() as conn:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    if user and check_password_hash(user['password_hash'], data['password']):
        session['user_id'] = user['id']
        session['username'] = user['username']
        session.permanent = True
        return jsonify({'message': 'Login successful', 'username': user['username']})
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/logout', methods=['POST'])
@log_request_decorator
def logout():
    session.clear()
    return jsonify({'message': 'Logged out successfully'})


@app.route('/api/session', methods=['GET'])
def check_session():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'username': session.get('username')})
    return jsonify({'authenticated': False})


@app.route('/api/consult', methods=['POST'])
@log_request_decorator
@auth_required
@rate_limit(max_requests=20, window=60)
@validate_input({
    'symptoms': {'required': True, 'type': 'list'},
    'age': {'required': False, 'type': 'int'},
    'gender': {'required': False, 'type': 'str'},
    'conditions': {'required': False, 'type': 'list'},
    'vitals': {'required': False, 'type': 'str'},
    'consultation_type': {'required': False, 'type': 'str'}
})
def consult():
    data = g.validated_data
    symptoms = [sanitize(s) for s in data['symptoms'] if s]
    if not symptoms:
        return jsonify({'error': 'Please provide at least one symptom'}), 400

    age = data.get('age', 30)
    gender = sanitize(data.get('gender', 'unknown'))
    conditions = [sanitize(c) for c in data.get('conditions', [])]
    vitals = sanitize(data.get('vitals', ''))
    consultation_type = sanitize(data.get('consultation_type', 'general'))

    symptom_vector = symptoms_to_vector(symptoms)

    if sum(symptom_vector) == 0:
        return jsonify({'error': 'None of the provided symptoms could be recognized. Please try different symptom descriptions.'}), 400

    result = predict(symptom_vector, age, gender, conditions)
    if not result:
        return jsonify({'error': 'Analysis engine is initializing. Please try again in a moment.'}), 503

    try:
        with get_db() as conn:
            conn.execute(
                '''INSERT INTO consultations 
                   (user_id, symptoms, age, gender, conditions, vitals, consultation_type,
                    risk_level, probability_score, severity_index, cluster_id, confidence_score,
                    emergency_flag, advisory, predicted_disease)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (
                    session['user_id'],
                    json.dumps(symptoms),
                    age, gender,
                    json.dumps(conditions),
                    vitals,
                    consultation_type,
                    result['risk_level'],
                    result['top_diseases'][0]['probability'] if result['top_diseases'] else 0,
                    result['severity_index'],
                    result['cluster_id'],
                    result['confidence_score'],
                    1 if result['emergency_flag'] else 0,
                    result['advisory_text'],
                    result['top_diseases'][0]['disease'] if result['top_diseases'] else 'Unknown'
                )
            )
    except Exception as e:
        logger.error(f"Failed to store consultation: {e}")

    return jsonify({
        'analysis': result,
        'input_summary': {
            'symptoms': symptoms,
            'age': age,
            'gender': gender,
            'conditions': conditions,
            'consultation_type': consultation_type
        },
        'timestamp': datetime.now().isoformat(),
        'disclaimer': 'This analysis is generated by an AI system for informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.'
    })


@app.route('/api/history', methods=['GET'])
@log_request_decorator
@auth_required
def get_history():
    with get_db() as conn:
        rows = conn.execute(
            '''SELECT id, symptoms, age, gender, risk_level, probability_score, 
                      severity_index, cluster_id, confidence_score, emergency_flag,
                      advisory, predicted_disease, created_at
               FROM consultations WHERE user_id = ? ORDER BY created_at DESC LIMIT 50''',
            (session['user_id'],)
        ).fetchall()

    history = []
    for row in rows:
        history.append({
            'id': row['id'],
            'symptoms': json.loads(row['symptoms']),
            'age': row['age'],
            'gender': row['gender'],
            'risk_level': row['risk_level'],
            'probability_score': row['probability_score'],
            'severity_index': row['severity_index'],
            'cluster_id': row['cluster_id'],
            'confidence_score': row['confidence_score'],
            'emergency_flag': bool(row['emergency_flag']),
            'advisory': row['advisory'],
            'predicted_disease': row['predicted_disease'],
            'created_at': row['created_at']
        })
    return jsonify({'history': history})


@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    formatted = [s.replace('_', ' ').title() for s in MASTER_SYMPTOM_LIST]
    return jsonify({'symptoms': formatted, 'raw': MASTER_SYMPTOM_LIST})


@app.route('/api/scrape-status', methods=['GET'])
@auth_required
def scrape_status():
    with get_db() as conn:
        count = conn.execute('SELECT COUNT(*) as cnt FROM scraped_data').fetchone()['cnt']
        latest = conn.execute('SELECT scraped_at FROM scraped_data ORDER BY scraped_at DESC LIMIT 1').fetchone()
    return jsonify({
        'total_records': count,
        'last_scraped': latest['scraped_at'] if latest else None,
        'models_trained': ML_MODELS['trained'],
        'last_trained': ML_MODELS['last_trained'].isoformat() if ML_MODELS['last_trained'] else None,
        'disease_count': len(ML_MODELS['disease_names'])
    })


@app.route('/api/retrain', methods=['POST'])
@auth_required
@rate_limit(max_requests=3, window=300)
def retrain():
    global SCRAPE_CACHE
    SCRAPE_CACHE = {}
    threading.Thread(target=scrape_medical_data, daemon=True).start()
    success = train_models()
    if success:
        return jsonify({'message': 'Models retrained successfully', 'disease_count': len(ML_MODELS['disease_names'])})
    return jsonify({'error': 'Retraining failed'}), 500


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    return response


def initialize():
    init_db()
    logger.info("Database initialized")
    threading.Thread(target=scrape_medical_data, daemon=True).start()
    threading.Thread(target=train_models, daemon=True).start()
    logger.info("Background scraping and model training started")


initialize()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
