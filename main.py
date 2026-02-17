"""
SENTINEL AI RESEARCH - Production Grade Investment Analysis Platform
Real-time market data with decision-oriented insights investors can't find elsewhere
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging
import requests
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentinel AI Research", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")

client = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except:
        pass

# Top 100 US stocks
TOP_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc. (Google)"},
    {"symbol": "AMZN", "name": "Amazon.com Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "META", "name": "Meta Platforms (Facebook)"},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "BRK.B", "name": "Berkshire Hathaway"},
    {"symbol": "JPM", "name": "JPMorgan Chase"},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "WMT", "name": "Walmart"},
    {"symbol": "JNJ", "name": "Johnson & Johnson"},
    {"symbol": "MA", "name": "Mastercard"},
    {"symbol": "PG", "name": "Procter & Gamble"},
    {"symbol": "HD", "name": "Home Depot"},
    {"symbol": "BAC", "name": "Bank of America"},
    {"symbol": "DIS", "name": "Walt Disney"},
    {"symbol": "NFLX", "name": "Netflix"},
    {"symbol": "KO", "name": "Coca-Cola"},
    {"symbol": "PEP", "name": "PepsiCo"}
]

class AnalysisRequest(BaseModel):
    company_name: str
    email: EmailStr

class AnalysisResponse(BaseModel):
    success: bool
    live_data: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def fetch_live_data(ticker: str) -> Dict[str, Any]:
    """Fetch real-time stock data from Alpha Vantage"""
    try:
        ticker = ticker.strip().upper()
        logger.info(f"🔍 Fetching live data for {ticker}")
        
        # Get real-time quote
        quote_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        quote_resp = requests.get(quote_url, timeout=10)
        quote_data = quote_resp.json()
        
        global_quote = quote_data.get("Global Quote", {})
        if not global_quote:
            raise ValueError("No live data available")
        
        time.sleep(0.5)  # Rate limit
        
        # Get company overview
        overview_url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        overview_resp = requests.get(overview_url, timeout=10)
        overview = overview_resp.json()
        
        # Parse live data
        current_price = float(global_quote.get("05. price", 0))
        change = float(global_quote.get("09. change", 0))
        change_pct = float(global_quote.get("10. change percent", "0").replace("%", ""))
        
        # Financial metrics
        pe = float(overview.get("PERatio", 0)) or None
        pb = float(overview.get("PriceToBookRatio", 0)) or None
        profit_margin = float(overview.get("ProfitMargin", 0)) * 100 if overview.get("ProfitMargin") else None
        roe = float(overview.get("ReturnOnEquityTTM", 0)) * 100 if overview.get("ReturnOnEquityTTM") else None
        div_yield = float(overview.get("DividendYield", 0)) * 100 if overview.get("DividendYield") else 0
        revenue_growth = float(overview.get("QuarterlyRevenueGrowthYOY", 0)) * 100 if overview.get("QuarterlyRevenueGrowthYOY") else None
        debt_equity = float(overview.get("DebtToEquity", 0)) or None
        beta = float(overview.get("Beta", 1)) or 1.0
        current_ratio = float(overview.get("CurrentRatio", 0)) or None
        
        # 52-week range
        week52_high = float(overview.get("52WeekHigh", current_price * 1.2))
        week52_low = float(overview.get("52WeekLow", current_price * 0.8))
        
        # Calculate position in range
        price_position = ((current_price - week52_low) / (week52_high - week52_low)) * 100 if week52_high != week52_low else 50
        
        # Calculate strategic levels
        buy_zone = round(current_price * 0.95, 2)
        take_profit_1 = round(current_price * 1.10, 2)
        take_profit_2 = round(current_price * 1.20, 2)
        stop_loss = round(current_price * 0.90, 2)
        
        result = {
            "ticker": ticker,
            "company_name": overview.get("Name", ticker),
            "sector": overview.get("Sector", "Unknown"),
            "industry": overview.get("Industry", "Unknown"),
            "exchange": overview.get("Exchange", "NASDAQ"),
            
            # Live pricing
            "current_price": round(current_price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 2),
            "currency": "USD",
            "last_updated": datetime.now().strftime("%B %d, %Y at %I:%M %p UTC"),
            
            # Valuation
            "market_cap": int(overview.get("MarketCapitalization", 0)),
            "pe_ratio": round(pe, 2) if pe else None,
            "pb_ratio": round(pb, 2) if pb else None,
            
            # Profitability
            "profit_margin": round(profit_margin, 2) if profit_margin else None,
            "roe": round(roe, 2) if roe else None,
            "dividend_yield": round(div_yield, 2),
            
            # Growth & Risk
            "revenue_growth": round(revenue_growth, 2) if revenue_growth else None,
            "debt_to_equity": round(debt_equity, 2) if debt_equity else None,
            "beta": round(beta, 2),
            "current_ratio": round(current_ratio, 2) if current_ratio else None,
            
            # Price levels
            "52_week_high": round(week52_high, 2),
            "52_week_low": round(week52_low, 2),
            "price_position_percent": round(price_position, 1),
            
            # Strategic levels
            "buy_zone_price": buy_zone,
            "take_profit_1_price": take_profit_1,
            "take_profit_2_price": take_profit_2,
            "stop_loss_price": stop_loss,
        }
        
        logger.info(f"✅ Live data fetched for {ticker}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise ValueError(f"Could not fetch live data for {ticker}. Please verify ticker symbol.")

def generate_comprehensive_analysis(data: Dict) -> Dict[str, Any]:
    """Generate comprehensive investment analysis with unique insights"""
    
    ticker = data['ticker']
    price = data['current_price']
    
    # ===== KEY DECISION POINTS =====
    decision_points = []
    
    # Price Action
    if data['change_percent'] > 2:
        decision_points.append(f"Price Action: Currently ${price} with strong upward momentum (+{data['change_percent']}% today)")
    elif data['change_percent'] < -2:
        decision_points.append(f"Price Action: Currently ${price} with negative daily momentum ({data['change_percent']}% today)")
    else:
        decision_points.append(f"Price Action: Currently ${price} with moderate daily change ({data['change_percent']:+.2f}%)")
    
    # Valuation Assessment
    pe = data.get('pe_ratio')
    if pe:
        if pe < 15:
            decision_points.append(f"Valuation Assessment: P/E ratio of {pe} indicates attractive valuation")
        elif pe > 30:
            decision_points.append(f"Valuation Assessment: P/E ratio of {pe} suggests premium pricing")
        else:
            decision_points.append(f"Valuation Assessment: P/E ratio of {pe} shows fair market valuation")
    
    # Market Position
    market_cap = data['market_cap']
    if market_cap > 200_000_000_000:
        cap_desc = f"${market_cap/1_000_000_000:.2f}B"
        decision_points.append(f"Market Position: {cap_desc} market cap - classified as large-cap stock")
    elif market_cap > 10_000_000_000:
        cap_desc = f"${market_cap/1_000_000_000:.2f}B"
        decision_points.append(f"Market Position: {cap_desc} market cap - classified as mid-cap stock")
    else:
        cap_desc = f"${market_cap/1_000_000:.2f}M"
        decision_points.append(f"Market Position: {cap_desc} market cap - classified as small-cap stock")
    
    # Sector Exposure
    decision_points.append(f"Sector Exposure: {data['sector']} sector, {data['industry']} industry")
    
    # Price Range
    decision_points.append(f"Price Range: Trading {data['price_position_percent']:.0f}% from 52-week low")
    
    # ===== LIVE VALUATION ANALYSIS =====
    valuation_metrics = []
    
    # P/E Ratio
    if pe:
        if pe < 15:
            status = "ATTRACTIVE"
            color = "green"
            explanation = f"You pay ${pe:.2f} for every $1 of earnings. Below market average - good value!"
        elif pe > 30:
            status = "EXPENSIVE"
            color = "red"
            explanation = f"You pay ${pe:.2f} for every $1 of earnings. Make sure growth justifies this premium!"
        else:
            status = "FAIR"
            color = "yellow"
            explanation = f"You pay ${pe:.2f} for every $1 of earnings. Market average valuation."
        
        valuation_metrics.append({
            "icon": "📊",
            "name": "P/E Ratio",
            "description": "Price-to-Earnings",
            "value": f"{pe:.2f}",
            "status": status,
            "color": color,
            "explanation": explanation
        })
    
    # P/B Ratio
    pb = data.get('pb_ratio')
    if pb:
        if pb > 10:
            explanation = f"Premium to Assets: Stock trades at {pb:.2f}x book value. Investors are paying for intangibles like brand and growth potential."
        else:
            explanation = f"Stock trades at {pb:.2f}x book value. Reasonable premium to assets."
        
        valuation_metrics.append({
            "icon": "📚",
            "name": "P/B Ratio",
            "description": "Price-to-Book",
            "value": f"{pb:.2f}",
            "status": "Premium" if pb > 5 else "Fair",
            "color": "yellow" if pb > 5 else "green",
            "explanation": explanation
        })
    
    # Profit Margin
    margin = data.get('profit_margin')
    if margin:
        if margin > 20:
            status = "EXCELLENT"
            color = "green"
            explanation = f"Company keeps ${margin:.2f} profit from every $100 in sales. Strong pricing power!"
        elif margin > 10:
            status = "GOOD"
            color = "yellow"
            explanation = f"Company keeps ${margin:.2f} profit from every $100 in sales. Solid profitability."
        else:
            status = "WEAK"
            color = "red"
            explanation = f"Company keeps only ${margin:.2f} profit from every $100 in sales. Thin margins."
        
        valuation_metrics.append({
            "icon": "💰",
            "name": "Profit Margin",
            "description": "How much profit per dollar",
            "value": f"{margin:.2f}%",
            "status": status,
            "color": color,
            "explanation": explanation
        })
    
    # ROE
    roe = data.get('roe')
    if roe:
        if roe > 15:
            status = "HIGH RETURNS"
            color = "green"
            explanation = f"Company generates ${roe:.2f} profit for every $100 invested by shareholders. Efficient use of capital!"
        elif roe > 10:
            status = "GOOD"
            color = "yellow"
            explanation = f"Company generates ${roe:.2f} profit for every $100 invested. Decent returns."
        else:
            status = "LOW"
            color = "red"
            explanation = f"Company generates ${roe:.2f} profit for every $100 invested. Below average efficiency."
        
        valuation_metrics.append({
            "icon": "🎯",
            "name": "ROE",
            "description": "Return on Equity",
            "value": f"{roe:.2f}%",
            "status": status,
            "color": color,
            "explanation": explanation
        })
    
    # Dividend Yield
    div = data.get('dividend_yield', 0)
    if div > 0:
        if div > 4:
            status = "HIGH INCOME"
            explanation = f"You earn ${div:.0f} per year for every $100 invested. Good for income-seeking investors!"
        elif div > 2:
            status = "INCOME"
            explanation = f"You earn ${div:.0f} per year for every $100 invested. Moderate income stream."
        else:
            status = "LOW INCOME"
            explanation = f"You earn ${div:.0f} per year for every $100 invested. Focus is on growth, not income."
        
        valuation_metrics.append({
            "icon": "💵",
            "name": "Dividend Yield",
            "description": "Income you receive",
            "value": f"{div:.0f}%",
            "status": status,
            "color": "green" if div > 3 else "yellow",
            "explanation": explanation
        })
    
    # ===== STRATEGIC ENTRY & EXIT LEVELS =====
    entry_exit_levels = {
        "buy_zone": {
            "level": "BUY ZONE",
            "price": f"Below ${data['buy_zone_price']}",
            "color": "green",
            "recommendation": "Strong accumulation opportunity - consider building position",
            "gain_potential": "Entry at discount"
        },
        "current": {
            "level": "CURRENT PRICE",
            "price": f"${price}",
            "color": "blue",
            "recommendation": "Monitor for confirmation signals before adding",
            "gain_potential": "Current market price"
        },
        "profit_1": {
            "level": "TAKE PROFIT 1",
            "price": f"Above ${data['take_profit_1_price']}",
            "color": "yellow",
            "recommendation": f"Consider booking 30-50% profits at +{((data['take_profit_1_price']/price - 1) * 100):.0f}% gain",
            "gain_potential": f"+{((data['take_profit_1_price']/price - 1) * 100):.0f}% gain"
        },
        "profit_2": {
            "level": "TAKE PROFIT 2",
            "price": f"Above ${data['take_profit_2_price']}",
            "color": "yellow",
            "recommendation": f"Exit remaining position at +{((data['take_profit_2_price']/price - 1) * 100):.0f}% for maximum gains",
            "gain_potential": f"+{((data['take_profit_2_price']/price - 1) * 100):.0f}% gain"
        },
        "stop_loss": {
            "level": "STOP LOSS",
            "price": f"Below ${data['stop_loss_price']}",
            "color": "red",
            "recommendation": "Strict exit level to limit downside risk to 10%",
            "gain_potential": "-10% risk limit"
        }
    }
    
    # ===== COMPREHENSIVE RISK ANALYSIS =====
    risk_factors = []
    
    # Price Positioning Risk
    pos_pct = data['price_position_percent']
    if pos_pct > 90:
        risk_factors.append({
            "category": "Price Positioning",
            "metric": f"{pos_pct:.1f}% of 52-week high",
            "assessment": "⚠ Near peak - limited upside, watch for reversal"
        })
    elif pos_pct < 30:
        risk_factors.append({
            "category": "Price Positioning",
            "metric": f"{pos_pct:.1f}% of 52-week high",
            "assessment": "✓ Significant discount - potential value opportunity"
        })
    else:
        risk_factors.append({
            "category": "Price Positioning",
            "metric": f"{pos_pct:.1f}% of 52-week high",
            "assessment": "○ Mid-range pricing - balanced risk/reward"
        })
    
    # Financial Leverage Risk
    debt = data.get('debt_to_equity')
    if debt:
        if debt > 100:
            risk_factors.append({
                "category": "Financial Leverage",
                "metric": f"Debt/Equity: {debt:.2f}",
                "assessment": "⚠ High leverage - monitor debt servicing ability"
            })
        elif debt < 50:
            risk_factors.append({
                "category": "Financial Leverage",
                "metric": f"Debt/Equity: {debt:.2f}",
                "assessment": "✓ Low debt - strong financial flexibility"
            })
        else:
            risk_factors.append({
                "category": "Financial Leverage",
                "metric": f"Debt/Equity: {debt:.2f}",
                "assessment": "○ Moderate leverage - industry standard"
            })
    
    # Liquidity Risk
    current_ratio = data.get('current_ratio')
    if current_ratio:
        if current_ratio < 1:
            risk_factors.append({
                "category": "Liquidity Position",
                "metric": f"Current Ratio: {current_ratio:.2f}",
                "assessment": "⚠ Liquidity concerns - monitor cash flow closely"
            })
        elif current_ratio > 2:
            risk_factors.append({
                "category": "Liquidity Position",
                "metric": f"Current Ratio: {current_ratio:.2f}",
                "assessment": "✓ Strong liquidity - can meet short-term obligations"
            })
        else:
            risk_factors.append({
                "category": "Liquidity Position",
                "metric": f"Current Ratio: {current_ratio:.2f}",
                "assessment": "○ Adequate liquidity - normal working capital"
            })
    
    # Volatility Risk
    beta = data.get('beta', 1)
    if beta > 1.3:
        risk_factors.append({
            "category": "Volatility (Beta)",
            "metric": f"{beta:.2f}",
            "assessment": "⚠ High volatility - expect larger price swings than market"
        })
    elif beta < 0.8:
        risk_factors.append({
            "category": "Volatility (Beta)",
            "metric": f"{beta:.2f}",
            "assessment": "✓ Lower volatility - more stable than overall market"
        })
    else:
        risk_factors.append({
            "category": "Volatility (Beta)",
            "metric": f"{beta:.2f}",
            "assessment": "↔ Average market volatility - moves with market"
        })
    
    # Sector Dynamics
    risk_factors.append({
        "category": "Sector Dynamics",
        "metric": data['sector'],
        "assessment": "Monitor sector-specific trends and regulatory changes"
    })
    
    return {
        "key_decision_points": decision_points,
        "valuation_analysis": valuation_metrics,
        "entry_exit_levels": entry_exit_levels,
        "risk_analysis": risk_factors
    }

@app.get("/")
async def root():
    return {
        "service": "Sentinel AI Research",
        "tagline": "Agentic AI Investment Intelligence",
        "version": "1.0.0",
        "status": "operational",
        "features": ["Live Market Data", "Agentic AI", "Actionable Insights"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/stocks")
async def get_stocks():
    """Get popular stocks for autocomplete"""
    return {"stocks": TOP_STOCKS}

@app.post("/api/generate-report", response_model=AnalysisResponse)
async def generate_report(request: AnalysisRequest):
    """Generate comprehensive investment analysis report"""
    try:
        ticker = request.company_name.strip().upper()
        logger.info(f"📊 Analyzing {ticker} for {request.email}")
        
        try:
            live_data = fetch_live_data(ticker)
        except Exception as e:
            return AnalysisResponse(success=False, error=str(e))
        
        analysis = generate_comprehensive_analysis(live_data)
        
        logger.info(f"✅ Analysis complete for {ticker}")
        
        return AnalysisResponse(
            success=True,
            live_data=live_data,
            analysis=analysis
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return AnalysisResponse(success=False, error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
