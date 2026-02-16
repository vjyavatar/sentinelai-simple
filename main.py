"""Sentinel AI Stock Analysis - Production Ready"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import yfinance as yf
import os
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentinel AI", version="1.0.0")

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
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, max_retries=2, timeout=60.0)
        logger.info("✅ AI client initialized")
    except Exception as e:
        logger.error(f"AI init error: {e}")

class AnalysisRequest(BaseModel):
    company_name: str
    email: EmailStr

class AnalysisResponse(BaseModel):
    success: bool
    live_data: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    error: Optional[str] = None

def fetch_data(ticker: str) -> Dict[str, Any]:
    """Fetch stock data with robust error handling"""
    try:
        logger.info(f"🔍 Fetching {ticker}")
        
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get info - this is the main data source
        try:
            info = stock.info
            if not info or len(info) < 5:
                # If info is empty or too small, try alternative
                raise ValueError("Info data incomplete")
        except Exception as e:
            logger.warning(f"Info fetch failed, trying history: {e}")
            # Fallback to historical data
            hist = stock.history(period="5d")
            if hist.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Build minimal info from history
            info = {
                "longName": ticker,
                "sector": "Unknown",
                "currency": "USD"
            }
        
        # Get historical data for price
        hist = stock.history(period="1mo")
        if hist.empty:
            raise ValueError(f"No price data for {ticker}")
        
        current_price = float(hist['Close'].iloc[-1])
        
        # Safely extract data with fallbacks
        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", info.get("shortName", ticker)),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "current_price": round(current_price, 2),
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": round(float(info.get("trailingPE", 0)), 2) if info.get("trailingPE") else None,
            "forward_pe": round(float(info.get("forwardPE", 0)), 2) if info.get("forwardPE") else None,
            "profit_margin": round(float(info.get("profitMargins", 0)) * 100, 2) if info.get("profitMargins") else None,
            "revenue_growth": round(float(info.get("revenueGrowth", 0)) * 100, 2) if info.get("revenueGrowth") else None,
            "debt_to_equity": round(float(info.get("debtToEquity", 0)), 2) if info.get("debtToEquity") else None,
            "52_week_high": round(float(info.get("fiftyTwoWeekHigh", current_price)), 2),
            "52_week_low": round(float(info.get("fiftyTwoWeekLow", current_price)), 2),
            "beta": round(float(info.get("beta", 1)), 2) if info.get("beta") else None,
            "dividend_yield": round(float(info.get("dividendYield", 0)) * 100, 2) if info.get("dividendYield") else 0,
        }
        
        logger.info(f"✅ Data fetched for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error fetching {ticker}: {str(e)}")
        raise ValueError(f"Could not fetch data for {ticker}. Error: {str(e)}")

def analyze(ticker: str, data: Dict) -> str:
    """Generate AI analysis"""
    if not client:
        logger.warning("AI unavailable, using fallback")
        return generate_fallback(ticker, data)
    
    try:
        logger.info(f"🧠 Generating AI analysis for {ticker}")
        
        prompt = f"""Analyze this stock and provide clear investment guidance:

**{ticker} - {data['company_name']}**
Sector: {data['sector']}
Current Price: ${data['current_price']} {data['currency']}

Financial Metrics:
- P/E Ratio: {data['pe_ratio']}
- Forward P/E: {data['forward_pe']}
- Profit Margin: {data['profit_margin']}%
- Revenue Growth: {data['revenue_growth']}%
- Debt/Equity: {data['debt_to_equity']}
- 52-Week Range: ${data['52_week_low']} - ${data['52_week_high']}
- Beta: {data['beta']}
- Dividend Yield: {data['dividend_yield']}%

Provide:
1. **VALUATION VERDICT**: Is it overvalued, fairly valued, or undervalued?
2. **FINANCIAL HEALTH**: Comment on profitability, growth, and debt levels
3. **RISK ASSESSMENT**: What are the main risks?
4. **FINAL RECOMMENDATION**: Clear BUY / HOLD / SELL with confidence level
5. **ENTRY STRATEGY**: If buying, what's a good entry price?

Be specific, data-driven, and actionable. Keep it professional but conversational."""

        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = msg.content[0].text
        logger.info(f"✅ AI analysis complete for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"AI error: {e}")
        return generate_fallback(ticker, data)

def generate_fallback(ticker: str, data: Dict) -> str:
    """Generate basic analysis when AI unavailable"""
    pe = data.get('pe_ratio', 0)
    margin = data.get('profit_margin', 0)
    growth = data.get('revenue_growth', 0)
    
    # Simple logic
    score = 0
    if pe and pe < 25: score += 1
    if margin and margin > 15: score += 1
    if growth and growth > 10: score += 1
    
    if score >= 2:
        rec = "BUY"
    elif score == 1:
        rec = "HOLD"
    else:
        rec = "SELL"
    
    return f"""**ANALYSIS FOR {ticker}**

**VALUATION**
P/E Ratio: {pe} - {"Attractive" if pe and pe < 25 else "Premium"} valuation
Current Price: ${data['current_price']}

**FINANCIAL HEALTH**
Profit Margin: {margin}% - {"Strong" if margin and margin > 15 else "Moderate"} profitability
Revenue Growth: {growth}% - {"Impressive" if growth and growth > 10 else "Steady"} growth

**RECOMMENDATION: {rec}**
Based on current metrics, this stock appears {"undervalued" if rec=="BUY" else "fairly valued" if rec=="HOLD" else "overvalued"}.

*Note: Full AI-powered analysis requires ANTHROPIC_API_KEY configuration.*"""

@app.get("/")
async def root():
    return {
        "service": "Sentinel AI Stock Analysis",
        "status": "operational",
        "version": "1.0.0",
        "ai_enabled": bool(client)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "ai_available": bool(client)
    }

@app.post("/api/generate-report", response_model=AnalysisResponse)
async def generate_report(request: AnalysisRequest):
    """Generate comprehensive stock analysis"""
    try:
        ticker = request.company_name.strip().upper()
        logger.info(f"📊 Request: {ticker} from {request.email}")
        
        # Fetch data
        try:
            live_data = fetch_data(ticker)
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return AnalysisResponse(
                success=False,
                error=str(e)
            )
        
        # Generate analysis
        try:
            report = analyze(ticker, live_data)
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            report = f"Basic data for {ticker}: ${live_data['current_price']}"
        
        logger.info(f"✅ Complete for {ticker}")
        
        return AnalysisResponse(
            success=True,
            live_data=live_data,
            report=report
        )
        
    except Exception as e:
        logger.error(f"❌ Request failed: {e}")
        return AnalysisResponse(
            success=False,
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
