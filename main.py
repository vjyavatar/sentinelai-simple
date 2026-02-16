"""
Sentinel AI Stock Analysis - Python FastAPI Service
Fixed version without proxy issues
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import yfinance as yf
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Sentinel AI Stock Analysis",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Anthropic client - fix for proxy issue
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = None

if ANTHROPIC_API_KEY:
    try:
        import anthropic
        # Create client without any proxy configuration
        anthropic_client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            max_retries=2,
            timeout=60.0
        )
        logger.info("✅ Anthropic client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Anthropic: {e}")
        anthropic_client = None


# Models
class AnalysisRequest(BaseModel):
    company_name: str
    email: EmailStr


class AnalysisResponse(BaseModel):
    success: bool
    live_data: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    error: Optional[str] = None


def fetch_stock_data(ticker: str) -> Dict[str, Any]:
    """Fetch real-time stock data"""
    try:
        logger.info(f"📊 Fetching data for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise ValueError(f"No data found for {ticker}")
        
        current_price = hist['Close'].iloc[-1]
        year_high = hist['High'].max()
        year_low = hist['Low'].min()
        
        data = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "current_price": round(current_price, 2),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
            "profit_margin": round(info.get("profitMargins", 0) * 100, 2) if info.get("profitMargins") else None,
            "roe": round(info.get("returnOnEquity", 0) * 100, 2) if info.get("returnOnEquity") else None,
            "52_week_high": round(year_high, 2),
            "52_week_low": round(year_low, 2),
            "beta": round(info.get("beta", 1), 2) if info.get("beta") else None,
        }
        
        logger.info(f"✅ Data fetched successfully for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"❌ Error fetching data: {e}")
        raise


def generate_analysis(ticker: str, data: Dict[str, Any]) -> str:
    """Generate AI analysis"""
    if not anthropic_client:
        logger.warning("⚠️ Anthropic client not available, using fallback")
        return generate_fallback_analysis(ticker, data)
    
    try:
        logger.info(f"🧠 Generating AI analysis for {ticker}")
        
        prompt = f"""Analyze this stock and provide investment insights:

{ticker} - {data.get('company_name')}
Sector: {data.get('sector')}
Current Price: ${data.get('current_price')}
P/E Ratio: {data.get('pe_ratio')}
Profit Margin: {data.get('profit_margin')}%
ROE: {data.get('roe')}%
52-Week Range: ${data.get('52_week_low')} - ${data.get('52_week_high')}

Provide:
1. Valuation Assessment
2. Financial Health
3. Recommendation (BUY/HOLD/SELL) with reasoning
Keep it concise and actionable."""

        message = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        analysis = message.content[0].text
        logger.info(f"✅ AI analysis generated for {ticker}")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ AI analysis error: {e}")
        return generate_fallback_analysis(ticker, data)


def generate_fallback_analysis(ticker: str, data: Dict[str, Any]) -> str:
    """Fallback analysis when AI unavailable"""
    pe = data.get('pe_ratio', 0)
    margin = data.get('profit_margin', 0)
    
    recommendation = "HOLD"
    if pe and pe < 20 and margin and margin > 15:
        recommendation = "BUY"
    elif pe and pe > 40 or (margin and margin < 5):
        recommendation = "SELL"
    
    return f"""ANALYSIS FOR {ticker}

Current Price: ${data.get('current_price')}
P/E Ratio: {pe} - {"Attractive" if pe and pe < 25 else "Premium"} valuation
Profit Margin: {margin}% - {"Strong" if margin and margin > 15 else "Moderate"} profitability

RECOMMENDATION: {recommendation}

Note: Full AI analysis requires ANTHROPIC_API_KEY configuration."""


# Endpoints
@app.get("/")
async def root():
    return {
        "service": "Sentinel AI Stock Analysis",
        "version": "1.0.0",
        "status": "operational",
        "ai_enabled": bool(anthropic_client)
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_available": bool(anthropic_client)
    }


@app.post("/api/generate-report", response_model=AnalysisResponse)
async def generate_report(request: AnalysisRequest):
    """Generate stock analysis report"""
    try:
        ticker = request.company_name.strip().upper()
        logger.info(f"📊 Analysis request for {ticker} from {request.email}")
        
        # Fetch data
        live_data = fetch_stock_data(ticker)
        
        # Generate analysis
        report = generate_analysis(ticker, live_data)
        
        logger.info(f"✅ Analysis complete for {ticker}")
        
        return AnalysisResponse(
            success=True,
            live_data=live_data,
            report=report
        )
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        return AnalysisResponse(
            success=False,
            error=str(e)
        )


@app.get("/api/verify-price/{ticker}")
async def verify_price(ticker: str):
    """Quick price verification"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        info = stock.info
        
        if hist.empty:
            return {"error": "Invalid ticker", "valid": False}
        
        return {
            "ticker": ticker.upper(),
            "price": round(hist['Close'].iloc[-1], 2),
            "company": info.get("longName", ticker),
            "valid": True
        }
    except Exception as e:
        return {"error": str(e), "valid": False}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
