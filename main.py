# File: app/main.py
import os
import pdfplumber
import spacy
import mysql.connector
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# --- MySQL connection ---
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="resume_ranker"
)
cursor = db.cursor()

# --- DB Setup (Run once manually) ---
# CREATE TABLE resumes (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), content TEXT);
# CREATE TABLE jobs (id INT AUTO_INCREMENT PRIMARY KEY, title VARCHAR(255), description TEXT);
# CREATE TABLE scores (resume_id INT, job_id INT, score FLOAT);

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())

def get_keywords(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]])

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    file = request.files['file']
    name = request.form['name']
    file_path = f"uploads/{file.filename}"
    file.save(file_path)
    text = extract_text_from_pdf(file_path)
    keywords = get_keywords(text)
    cursor.execute("INSERT INTO resumes (name, content) VALUES (%s, %s)", (name, keywords))
    db.commit()
    return jsonify({"message": "Resume uploaded successfully"})

@app.route('/upload_job', methods=['POST'])
def upload_job():
    title = request.form['title']
    description = request.form['description']
    keywords = get_keywords(description)
    cursor.execute("INSERT INTO jobs (title, description) VALUES (%s, %s)", (title, keywords))
    db.commit()
    return jsonify({"message": "Job uploaded successfully"})

@app.route('/match', methods=['POST'])
def match():
    cursor.execute("SELECT id, content FROM resumes")
    resumes = cursor.fetchall()
    cursor.execute("SELECT id, description FROM jobs")
    jobs = cursor.fetchall()

    for r_id, r_text in resumes:
        for j_id, j_text in jobs:
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform([r_text, j_text])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            cursor.execute("INSERT INTO scores (resume_id, job_id, score) VALUES (%s, %s, %s)", (r_id, j_id, score))
    db.commit()
    return jsonify({"message": "Matching complete"})

@app.route('/top_matches/<int:job_id>', methods=['GET'])
def top_matches(job_id):
    cursor.execute("SELECT resumes.name, scores.score FROM scores JOIN resumes ON scores.resume_id = resumes.id WHERE scores.job_id = %s ORDER BY scores.score DESC LIMIT 5", (job_id,))
    matches = cursor.fetchall()
    return jsonify([{"name": name, "score": score} for name, score in matches])

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
