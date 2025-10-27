import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    pipeline = pickle.load(f)

app = FastAPI()

class LeadData(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

@app.get("/")
def read_root():
    return {"message": "Lead Scoring API"}

@app.post("/predict")
def predict_lead_conversion(lead_data: LeadData):
    # Convert to dict for the pipeline
    record = {
        "lead_source": lead_data.lead_source,
        "number_of_courses_viewed": lead_data.number_of_courses_viewed,
        "annual_income": lead_data.annual_income
    }
    
    # Get probability
    probability = pipeline.predict_proba([record])[0][1]
    
    return {"probability": probability}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)