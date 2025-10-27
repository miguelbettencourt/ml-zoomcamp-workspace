#!/usr/bin/env python3

import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

# The record to score
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Score the record
probability = pipeline.predict_proba([record])[0][1]

print(f"Probability that this lead will convert: {probability:.3f}")
