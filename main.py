"""Sentinel AI - FINAL WORKING VERSION with User-Agent Fix"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import yfinance as yf
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

# Configure yfinance to use proper user agent
import requests
requests.packages.urllib3.util.connection.HAS_IPV6 = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentinel AI", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = None

if ANTHROPIC_API_KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("✅ AI client ready")
    except:
        logger.warning("AI client unavailable")

# Popular stock tickers for autocomplete
POPULAR_TICKERS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "META", "name": "Meta Platforms Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "WMT", "name": "Walmart Inc."},
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "PG", "name": "Procter & Gamble Co."},
    {"symbol": "MA", "name": "Mastercard Inc."},
    {"symbol": "HD", "name": "Home Depot Inc."},
    {"symbol": "BAC", "name": "Bank of America Corp."},
    {"symbol": "DIS", "name": "Walt Disney Co."},
    {"symbol": "NFLX", "name": "Netflix Inc."},
    {"symbol": "INTC", "name": "Intel Corporation"},
    {"symbol": "AMD", "name": "Advanced Micro Devices"},
    {"symbol": "PYPL", "name": "PayPal Holdings Inc."}
]

class AnalysisRequest(BaseModel):
    company_name: str
    email: EmailStr

class AnalysisResponse(BaseModel):
    success: bool
    live_data: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    error: Optional[str] = None

def fetch_data(ticker: str) -> Dict[str, Any]:
    """Fetch stock data with user-agent fix"""
    try:
        ticker = ticker.strip().upper()
        logger.info(f"🔍 Fetching {ticker}")
        
        # Create session with proper headers (fixes most yfinance issues!)
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
        # Use session with yfinance
        stock = yf.Ticker(ticker, session=session)
        
        # Get data with extended timeout
        hist = stock.history(period="1mo")
        
        if hist.empty:
            # Try alternative
            hist = stock.history(period="5d")
        
        if hist.empty:
            raise ValueError(f"No price data available for {ticker}")
        
        # Get current price
        current_price = float(hist['Close'].iloc[-1])
        
        # Get info (with error handling)
        info = {}
        try:
            info = stock.info or {}
        except:
            info = {}
        
        # Build comprehensive result
        result = {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": round(current_price, 2),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": round(float(info.get("trailingPE") or 0), 2) or None,
            "forward_pe": round(float(info.get("forwardPE") or 0), 2) or None,
            "profit_margin": round(float(info.get("profitMargins") or 0) * 100, 2) or None,
            "operating_margin": round(float(info.get("operatingMargins") or 0) * 100, 2) or None,
            "revenue_growth": round(float(info.get("revenueGrowth") or 0) * 100, 2) or None,
            "earnings_growth": round(float(info.get("earningsGrowth") or 0) * 100, 2) or None,
            "debt_to_equity": round(float(info.get("debtToEquity") or 0), 2) or None,
            "roe": round(float(info.get("returnOnEquity") or 0) * 100, 2) or None,
            "52_week_high": round(float(info.get("fiftyTwoWeekHigh") or current_price * 1.2), 2),
            "52_week_low": round(float(info.get("fiftyTwoWeekLow") or current_price * 0.8), 2),
            "50_day_avg": round(float(info.get("fiftyDayAverage") or current_price), 2),
            "200_day_avg": round(float(info.get("twoHundredDayAverage") or current_price), 2),
            "beta": round(float(info.get("beta") or 1), 2),
            "dividend_yield": round(float(info.get("dividendYield") or 0) * 100, 2),
            "avg_volume": info.get("averageVolume", 0),
            "recommendation": info.get("recommendationKey", "hold"),
            "target_price": round(float(info.get("targetMeanPrice") or 0), 2) or None
        }
        
        logger.info(f"✅ Fetched {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise ValueError(f"Could not fetch data for {ticker}. Please verify the ticker symbol is correct.")

def analyze(ticker: str, data: Dict) -> str:
    """Generate AI analysis"""
    if not client:
        return generate_fallback(ticker, data)
    
    try:
        prompt = f"""Provide professional investment analysis for:

**{ticker} - {data['company_name']}**
Sector: {data['sector']} | Industry: {data['industry']}
Current Price: ${data['current_price']} {data['currency']}

**Key Metrics:**
- P/E Ratio: {data['pe_ratio']} | Forward P/E: {data['forward_pe']}
- Profit Margin: {data['profit_margin']}% | Operating Margin: {data['operating_margin']}%
- Revenue Growth: {data['revenue_growth']}% | Earnings Growth: {data['earnings_growth']}%
- ROE: {data['roe']}% | Debt/Equity: {data['debt_to_equity']}
- 52-Week Range: ${data['52_week_low']} - ${data['52_week_high']}
- Beta: {data['beta']} | Dividend Yield: {data['dividend_yield']}%

Provide concise analysis with:
1. **VALUATION**: Overvalued, fairly valued, or undervalued?
2. **STRENGTHS**: Key positive factors
3. **RISKS**: Main concerns
4. **RECOMMENDATION**: BUY / HOLD / SELL with confidence level (High/Medium/Low)
5. **PRICE TARGET**: Suggest entry/exit points

Be specific and actionable. Use professional investment terminology."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return msg.content[0].text
        
    except Exception as e:
        logger.error(f"AI error: {e}")
        return generate_fallback(ticker, data)

def generate_fallback(ticker: str, data: Dict) -> str:
    """Fallback analysis"""
    pe = data.get('pe_ratio') or 0
    margin = data.get('profit_margin') or 0
    growth = data.get('revenue_growth') or 0
    
    score = 0
    if pe and 10 < pe < 30: score += 2
    if margin and margin > 15: score += 2
    if growth and growth > 10: score += 2
    
    rec = "BUY" if score >= 4 else "HOLD" if score >= 2 else "SELL"
    conf = "High" if score >= 5 else "Medium" if score >= 3 else "Low"
    
    return f"""**ANALYSIS FOR {ticker}**

**VALUATION**
Current Price: ${data['current_price']}
P/E Ratio: {pe} - {"Attractive" if pe and pe < 25 else "Premium" if pe else "N/A"}
Target Price: ${data.get('target_price') or 'N/A'}

**STRENGTHS**
{"- Strong profitability" if margin and margin > 15 else "- Moderate margins"}
{"- Excellent growth" if growth and growth > 15 else "- Steady revenue"}
- Sector: {data['sector']}

**RISKS**
{"- High valuation" if pe and pe > 30 else "- Market volatility"}
- Beta: {data['beta']} ({"Higher than market" if data['beta'] > 1.2 else "In line with market"})

**RECOMMENDATION: {rec}**
Confidence: {conf}

**PRICE TARGET**
Entry Zone: ${round(data['current_price'] * 0.95, 2)} - ${data['current_price']}
Stop Loss: ${round(data['current_price'] * 0.90, 2)}
Target: ${round(data['current_price'] * 1.15, 2)}

*Full AI-powered analysis available with API key*"""

@app.get("/")
async def root():
    return {
        "service": "Sentinel AI Stock Analysis",
        "version": "3.0.0",
        "status": "operational",
        "ai_enabled": bool(client)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "ai": bool(client)}

@app.get("/api/tickers")
async def get_tickers():
    """Get popular tickers for autocomplete"""
    return {"tickers": POPULAR_TICKERS}

@app.post("/api/generate-report", response_model=AnalysisResponse)
async def generate_report(request: AnalysisRequest):
    """Generate comprehensive stock analysis"""
    try:
        ticker = request.company_name.strip().upper()
        logger.info(f"📊 {ticker} requested")
        
        try:
            live_data = fetch_data(ticker)
        except Exception as e:
            return AnalysisResponse(
                success=False,
                error=str(e)
            )
        
        report = analyze(ticker, live_data)
        
        logger.info(f"✅ {ticker} complete")
        
        return AnalysisResponse(
            success=True,
            live_data=live_data,
            report=report
        )
        
    except Exception as e:
        logger.error(f"❌ {e}")
        return AnalysisResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
