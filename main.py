"""
SIMPLEST POSSIBLE VERSION - GUARANTEED TO WORK
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

# CORS - MUST HAVE THIS!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    company_name: str
    email: str

@app.get("/")
def root():
    return {"status": "working", "message": "Sentinel AI is running!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/stocks")
def get_stocks():
    stocks = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "TSLA", "name": "Tesla, Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
    ]
    return {"stocks": stocks}

@app.post("/api/generate-report")
def generate_report(request: Request):
    ticker = request.company_name.upper()
    
    # Generate random realistic data
    price = round(random.uniform(50, 500), 2)
    change = round(random.uniform(-5, 5), 2)
    change_pct = round((change / price) * 100, 2)
    
    return {
        "success": True,
        "live_data": {
            "ticker": ticker,
            "company_name": f"{ticker} Inc.",
            "sector": "Technology",
            "current_price": price,
            "change": change,
            "change_percent": change_pct,
            "pe_ratio": round(random.uniform(15, 35), 2),
            "profit_margin": round(random.uniform(10, 30), 2),
            "revenue_growth": round(random.uniform(-5, 25), 2),
            "data_source": "Demo Data"
        },
        "analysis": {
            "recommendation": "BUY" if random.random() > 0.5 else "HOLD",
            "score": random.randint(5, 9),
            "max_score": 10,
            "action": "buy",
            "buy_reasons": [
                f"Current price of ${price} presents opportunity",
                "Strong market position in sector",
                "Positive growth trajectory"
            ],
            "sell_reasons": []
        },
        "similar_stocks": ["AAPL", "MSFT", "GOOGL"]
    }

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
