Job Post Fraud Detection System with AI



The **Job Post Fraud Detection System with AI** is a web-based application designed to detect fraudulent job postings using Machine Learning and Natural Language Processing (NLP).

The system analyzes job descriptions, identifies suspicious patterns, and provides real-time alerts to help users avoid scams. It enhances trust and safety in online job marketplaces.

---

##  Features

### Core Functionalities

* **Job Submission** – Users can submit job posts for analysis
* **Fraud Detection Engine** – Uses ML + NLP to detect suspicious content
* **Real-Time Result** – Instant classification (Fraud / Legitimate)
* **Fraud Alerts** – Highlights risky phrases and patterns
* **Explainability Module** – Shows why a job is flagged
* **PDF Report Generation** – Downloadable analysis report
* **User Feedback System** – Collects user validation

---

### AI-Based Modules

* Semantic Analysis
* Duplicate Detection
* Suspicious Phrase Detection
* Sensitive Information Detection
* Geo & Language Consistency Checker
* External Verification
* Payment Risk Analysis
* Persuasion & Urgency Detection

---

##  Project Structure

```
JOB-FRAUD-DETECTION/
│
├── data/                     # Dataset files
├── models/                   # Trained ML models
├── static/                   # CSS, JS, assets
├── templates/               # HTML templates (Atomic Design)
│   ├── components/
│   │   ├── atoms/
│   │   ├── molecules/
│   │   ├── organisms/
│   │   └── templates/
│   └── pages/
│       ├── chat.html
│       ├── dashboard.html
│       ├── login.html
│       ├── register.html
│       ├── result.html
│       ├── report.html
│       └── submit_job.html
│
├── app.py                   # Main Flask application
├── model.py                 # ML model logic
├── anomaly_detector.py
├── duplicate_detector.py
├── semantic_analyzer.py
├── suspicious_phrases.py
├── sensitive_info_detector.py
├── geo_language_checker.py
├── external_verifier.py
├── payment_risk.py
├── persuasion_urgency.py
│
├── database.db              # SQLite database
├── create_db.py             # DB setup script
├── README.md
```

---

## Technologies Used

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **Database:** SQLite
* **AI/ML:** Scikit-learn, NLP techniques
* **Others:** PDF generation, API integration

---

##  Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/zihad-007/Job-Post-Fraud-Detection-System-with-AI-.git
cd job-fraud-detection
```

###  Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

###  Setup Database

```bash
python create_db.py
```

###  Run the Application

```bash
python app.py
```

### Open in Browser

```
http://127.0.0.1:5000/
```

---

## How It Works

1. User submits a job post
2. Text is preprocessed (cleaning, tokenization)
3. Multiple AI modules analyze the content
4. Risk scores are calculated
5. Final classification is generated
6. Results + explanations are shown

---

## Output

*  Fraud / Legitimate classification
*  Highlighted suspicious content
*  Risk analysis report
* Downloadable PDF report

---

## Security Considerations

* Input validation to prevent injection
* Sensitive data detection
* External verification checks
* Risk scoring for financial fraud patterns

---

## Future Enhancements (v2.0)

* Admin Dashboard
* Real-time API integration with job platforms
* Deep learning-based detection
* User reputation system
* Multi-language support
* Browser extension for job fraud detection

---

## Contribution

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## License

This project is for educational and research purposes.



## Acknowledgment

This project was developed to improve online job safety and reduce fraud using AI.
