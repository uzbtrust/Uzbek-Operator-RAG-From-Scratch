import argparse
import json
import random

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

QA_BANK = {
    "working_hours": {
        "contexts": [
            "Working hours: 9:00-18:00, Monday to Friday",
            "Our office is open from 8:00 AM to 6:00 PM on weekdays",
            "Business hours are Monday through Saturday, 10:00 to 20:00",
            "We are available 24/7 for emergency support",
            "Office hours: Mon-Fri 9:00-17:00, Sat 10:00-14:00",
        ],
        "questions": [
            "What are the working hours?",
            "When is the office open?",
            "What time do you open?",
            "What are your business hours?",
            "Are you open on weekends?",
            "When can I visit the office?",
        ],
    },
    "contact": {
        "contexts": [
            "Phone: +998 71 123 45 67. Email: info@operator.uz",
            "Contact us at +998 90 111 22 33 or support@company.com",
            "Hotline: 1099 (free). International: +998 71 200 00 01",
            "WhatsApp: +998 93 555 66 77. Telegram: @operator_support",
            "Call center: +998 71 999 88 77, available 24/7",
        ],
        "questions": [
            "What is your phone number?",
            "How can I contact you?",
            "What is the email address?",
            "Do you have a WhatsApp number?",
            "What is the hotline number?",
            "How do I reach customer support?",
        ],
    },
    "address": {
        "contexts": [
            "Address: 100100, Tashkent, Amir Temur street, 4",
            "Main office: Tashkent city, Yunusabad district, Bogishamol str. 220",
            "Branch offices in Samarkand, Bukhara, and Fergana",
            "Head office located at Navoi street 30, Tashkent 100011",
            "Visit us at: Chilanzar district, Bunyodkor avenue 1, Tashkent",
        ],
        "questions": [
            "What is your address?",
            "Where is the main office?",
            "Where are you located?",
            "Do you have branches in other cities?",
            "What is the office location?",
        ],
    },
    "pricing": {
        "contexts": [
            "Monthly subscription: Basic - 50,000 UZS, Standard - 100,000 UZS, Premium - 200,000 UZS",
            "Internet tariff: 10 Mbps for 80,000 UZS/month, 50 Mbps for 150,000 UZS/month",
            "Connection fee: 50,000 UZS one-time payment",
            "Roaming charges: incoming calls 500 UZS/min, outgoing 1500 UZS/min",
            "SMS bundle: 100 SMS for 5,000 UZS, 500 SMS for 20,000 UZS",
        ],
        "questions": [
            "How much does it cost?",
            "What are the prices?",
            "What are the tariff plans?",
            "How much is the monthly subscription?",
            "What is the connection fee?",
            "Do you have a premium plan?",
        ],
    },
    "services": {
        "contexts": [
            "Services: mobile communication, internet, TV, cloud storage",
            "We provide fiber optic internet, IPTV, and VoIP services",
            "Available services include: SIM card registration, bill payment, device repair",
            "Enterprise solutions: VPN, dedicated server, colocation, cloud hosting",
            "Additional services: call forwarding, voicemail, conference calling",
        ],
        "questions": [
            "What services do you offer?",
            "Do you provide internet services?",
            "Can I get a TV subscription?",
            "What enterprise solutions do you have?",
            "Do you offer cloud services?",
        ],
    },
    "documents": {
        "contexts": [
            "Required documents: passport or ID card, INPS (tax ID)",
            "For registration you need: original passport, 2 photos 3x4",
            "Corporate clients need: company registration certificate, director's passport, power of attorney",
            "To change tariff: visit office with passport or call hotline",
            "For SIM replacement: bring original passport to any service center",
        ],
        "questions": [
            "What documents do I need?",
            "What do I need for registration?",
            "What documents are required for corporate clients?",
            "How can I change my tariff plan?",
            "What do I need for SIM replacement?",
        ],
    },
    "complaints": {
        "contexts": [
            "File complaints via email: complaints@operator.uz or call 1099",
            "Complaint response time: 3 business days",
            "You can submit complaints online at my.operator.uz/feedback",
            "For billing disputes, contact billing@operator.uz with your account number",
            "Escalation: if not resolved in 5 days, contact regulatory@operator.uz",
        ],
        "questions": [
            "How do I file a complaint?",
            "What is the complaint response time?",
            "Where can I submit feedback?",
            "I have a billing issue, what should I do?",
            "How do I escalate a complaint?",
        ],
    },
    "payment": {
        "contexts": [
            "Payment methods: Payme, Click, cash at office, bank transfer",
            "Auto-payment available through Payme and Click apps",
            "Pay at any PAYNET terminal using your account number",
            "Credit card payments accepted: Visa, MasterCard, UzCard, Humo",
            "Bank transfer details: account 20208000123456789, MFO 00444, INN 123456789",
        ],
        "questions": [
            "How can I pay?",
            "What payment methods do you accept?",
            "Can I set up auto-payment?",
            "Do you accept credit cards?",
            "What are the bank transfer details?",
            "Can I pay at a terminal?",
        ],
    },
    "tech_support": {
        "contexts": [
            "Technical support: +998 71 200 00 02, available 24/7",
            "For internet issues: restart your router, wait 5 minutes, then call support",
            "Speed test at speedtest.operator.uz to check your connection",
            "Router configuration guide available at docs.operator.uz/setup",
            "On-site technical support available within 24 hours of request",
        ],
        "questions": [
            "My internet is not working, what should I do?",
            "How do I contact technical support?",
            "How can I test my internet speed?",
            "Where is the router setup guide?",
            "Can you send a technician?",
        ],
    },
    "promotions": {
        "contexts": [
            "Current promotion: 50% off first 3 months for new subscribers",
            "Refer a friend and get 1 month free for both",
            "Student discount: 30% off with valid student ID",
            "Bundle offer: Internet + TV for 120,000 UZS instead of 180,000 UZS",
            "Loyalty program: 5% cashback after 1 year, 10% after 3 years",
        ],
        "questions": [
            "Are there any promotions?",
            "Do you have student discounts?",
            "What is the referral program?",
            "Any bundle deals?",
            "Do you have a loyalty program?",
        ],
    },
}

UNANSWERABLE = [
    "What is the CEO's birthday?",
    "Do you sell smartphones?",
    "What color is the office building?",
    "Can I buy a laptop from you?",
    "What is the WiFi password?",
    "Do you have parking?",
    "What is the dress code?",
    "Do you serve coffee at the office?",
    "What programming languages do your developers use?",
    "How many employees do you have?",
    "What's the company revenue?",
    "Who founded the company?",
    "Do you sponsor any sports teams?",
    "What operating system do your servers use?",
    "Can I get a tour of the data center?",
]


def make_answer(question, context):
    return f"Based on our information: {context}"


def generate(n=1000):
    data = []
    cats = list(QA_BANK.keys())

    while len(data) < n:
        cat = random.choice(cats)
        q = random.choice(QA_BANK[cat]["questions"])
        ctx = random.choice(QA_BANK[cat]["contexts"])

        data.append({
            "question": q,
            "context": ctx,
            "answer": make_answer(q, ctx),
            "category": cat,
        })

    for q in UNANSWERABLE:
        data.append({
            "question": q,
            "context": "",
            "answer": "I don't have enough information to answer this question.",
            "category": "no_info",
        })

    random.shuffle(data)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/synthetic_qa.json")
    ap.add_argument("--num-pairs", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pairs = generate(args.num_pairs)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"{len(pairs)} ta QA juftlik yaratildi -> {args.output}")

    stats = {}
    for p in pairs:
        stats[p["category"]] = stats.get(p["category"], 0) + 1
    for k, v in sorted(stats.items()):
        logger.info(f"  {k}: {v}")
