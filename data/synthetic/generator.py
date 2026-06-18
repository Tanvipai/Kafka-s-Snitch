import csv
import os
import random
import string
from pathlib import Path

from faker import Faker
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

fake = Faker()
Faker.seed(42)
random.seed(42)

OUTPUT_DIR = Path(__file__).parent / "samples"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fake_ssn():
    """Format: XXX-XX-XXXX"""
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"


def fake_credit_card():
    return fake.credit_card_number(card_type="visa")


def fake_api_key():
    chars = string.ascii_uppercase + string.digits
    return "AKIA" + "".join(random.choices(chars, k=16))


def fake_password():
    chars = string.ascii_letters + string.digits + "!@#$%"
    return "".join(random.choices(chars, k=16))


def generate_hr_csv(index: int):

    file_path = OUTPUT_DIR / f"hr_employees_{index:03d}.csv"

    rows = []
    for _ in range(random.randint(20, 50)):
        rows.append({
            "employee_id": fake.uuid4()[:8].upper(),
            "full_name": fake.name(),
            "email": fake.company_email(),
            "phone": fake.phone_number(),
            "ssn": fake_ssn(),
            "department": random.choice(["Engineering", "HR", "Finance", "Legal", "Sales"]),
            "salary": random.randint(40000, 180000),
            "hire_date": fake.date_between(start_date="-10y", end_date="today").isoformat(),
            "manager": fake.name(),
        })

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  created: {file_path.name}")


def generate_medical_note(index: int):
    """Patient notes with diagnoses, medications, patient IDs."""
    file_path = OUTPUT_DIR / f"patient_note_{index:03d}.txt"

    diagnoses = [
        "Type 2 Diabetes Mellitus", "Hypertension", "Major Depressive Disorder",
        "Chronic Kidney Disease Stage 3", "Atrial Fibrillation", "Hypothyroidism",
        "Generalized Anxiety Disorder", "Osteoarthritis", "COPD", "Migraine"
    ]
    medications = [
        "Metformin 500mg", "Lisinopril 10mg", "Sertraline 50mg",
        "Atorvastatin 20mg", "Amlodipine 5mg", "Levothyroxine 75mcg"
    ]

    content = f"""PATIENT MEDICAL RECORD
======================
Patient ID   : MRN-{random.randint(100000, 999999)}
Patient Name : {fake.name()}
DOB          : {fake.date_of_birth(minimum_age=18, maximum_age=90).isoformat()}
SSN          : {fake_ssn()}
Insurance ID : INS-{random.randint(10000000, 99999999)}

Visit Date   : {fake.date_this_year().isoformat()}
Physician    : Dr. {fake.name()}
Department   : {random.choice(["Cardiology", "Endocrinology", "Psychiatry", "Nephrology", "General Practice"])}

PRIMARY DIAGNOSIS
-----------------
{random.choice(diagnoses)}

SECONDARY DIAGNOSIS
-------------------
{random.choice(diagnoses)}

CURRENT MEDICATIONS
-------------------
{chr(10).join(f"- {med}" for med in random.sample(medications, k=3))}

CLINICAL NOTES
--------------
{fake.paragraph(nb_sentences=6)}
Patient was advised to {random.choice(["follow up in 3 months", "schedule lab work", "monitor blood pressure daily", "return if symptoms worsen"])}.

Physician Signature: Dr. {fake.name()} | License: {random.randint(100000, 999999)}
"""

    file_path.write_text(content, encoding="utf-8")
    print(f"  created: {file_path.name}")


def generate_python_script(index: int):
    file_path = OUTPUT_DIR / f"config_script_{index:03d}.py"

    db_host = fake.ipv4()
    content = f"""#!/usr/bin/env python3
# Database configuration and API setup
# TODO: move these to environment variables (tech debt from v1)

import boto3
import psycopg2

# AWS credentials — DO NOT COMMIT
AWS_ACCESS_KEY_ID = "{fake_api_key()}"
AWS_SECRET_ACCESS_KEY = "{fake.sha256()[:40]}"
AWS_REGION = "us-east-1"

# Database connection
DB_CONFIG = {{
    "host": "{db_host}",
    "port": 5432,
    "database": "{fake.word()}_{random.choice(["prod", "staging", "dev"])}",
    "user": "{fake.user_name()}",
    "password": "{fake_password()}",
}}

#Third-party API keys
STRIPE_SECRET_KEY = "sk_live_{''.join(random.choices(string.ascii_letters + string.digits, k=32))}"
SENDGRID_API_KEY = "SG.{''.join(random.choices(string.ascii_letters + string.digits, k=40))}"
GITHUB_TOKEN = "ghp_{''.join(random.choices(string.ascii_letters + string.digits, k=36))}"


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
"""

    file_path.write_text(content, encoding="utf-8")
    print(f"  created: {file_path.name}")


def generate_financial_pdf(index: int):
    
    file_path = OUTPUT_DIR / f"invoice_{index:03d}.pdf"

    c = canvas.Canvas(str(file_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, f"INVOICE #{random.randint(10000, 99999)}")

    c.setFont("Helvetica", 11)
    y = height - 100

    lines = [
        f"Date: {fake.date_this_year().isoformat()}",
        f"Bill To: {fake.name()}",
        f"Email: {fake.email()}",
        f"Address: {fake.address().replace(chr(10), ', ')}",
        "",
        f"Amount Due: ${random.randint(100, 9999)}.{random.randint(0,99):02d}",
        f"Payment Method: Visa ending in {random.randint(1000,9999)}",
        f"Card Number: {fake_credit_card()}",
        f"Transaction ID: TXN-{fake.uuid4()[:12].upper()}",
        "",
        "Items:",
    ]

    for _ in range(random.randint(2, 5)):
        qty = random.randint(1, 10)
        price = random.uniform(10, 500)
        lines.append(f"  - {fake.bs().title()} x{qty} @ ${price:.2f}")

    for line in lines:
        c.drawString(50, y, line)
        y -= 20

    c.save()
    print(f"  created: {file_path.name}")


def generate_all(
    n_hr=15,
    n_medical=15,
    n_scripts=10,
    n_financial=10,
):
    ensure_output_dir()
    total = n_hr + n_medical + n_scripts + n_financial
    print(f"\ngenerating {total} synthetic files in {OUTPUT_DIR}\n")

    print("HR CSV files:")
    for i in range(n_hr):
        generate_hr_csv(i)

    print("\nMedical notes:")
    for i in range(n_medical):
        generate_medical_note(i)

    print("\nPython scripts with credentials:")
    for i in range(n_scripts):
        generate_python_script(i)

    print("\nFinancial PDFs:")
    for i in range(n_financial):
        generate_financial_pdf(i)

    print(f"\ndone. {total} files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    generate_all()