#app.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import asyncio
import os
from dotenv import load_dotenv
import pandas as pd
import requests
from newsapi import NewsApiClient
import time
import google.generativeai as genai
from alpha_vantage.timeseries import TimeSeries

# Load environment variables and initialize APIs
load_dotenv()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = st.secrets["NEWS_API_KEY"] if "NEWS_API_KEY" in st.secrets else os.getenv("NEWS_API_KEY")
BING_API_KEY = st.secrets["BING_API_KEY"] if "BING_API_KEY" in st.secrets else os.getenv("BING_API_KEY")
ALPHA_VANTAGE_KEY = st.secrets["ALPHA_VANTAGE_KEY"] if "ALPHA_VANTAGE_KEY" in st.secrets else os.getenv("ALPHA_VANTAGE_KEY")
YAHOO_FINANCE_KEY = st.secrets["YAHOO_FINANCE_KEY"] if "YAHOO_FINANCE_KEY" in st.secrets else os.getenv("YAHOO_FINANCE_KEY")

newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
BING_API_KEY = os.getenv("BING_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro')
alpha_vantage = TimeSeries(key=ALPHA_VANTAGE_KEY)

# Enhanced page config
st.set_page_config(
    layout="wide",
    page_title="AI Market Edge",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Improved CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .main {
        padding: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #6b7280;
        font-size: 0.875rem;
    }
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .chart-container {
        margin: 1.5rem 0;
        padding: 1rem;
        background: white;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@dataclass
class StartupInfo:
    name: str
    industry: str
    target_market: str
    target_demographics: str
    description: str
    goals: List[str]
    competitors: List[str]

class RateLimiter:
    """Rate limiter to prevent API throttling"""
    def __init__(self, calls_per_minute=50):
        self.calls_per_minute = calls_per_minute
        self.calls = []
        self.lock = asyncio.Lock()

    async def wait(self):
        """Wait if necessary to stay within rate limits"""
        async with self.lock:
            now = time.time()
            # Remove calls older than 1 minute
            self.calls = [call for call in self.calls if now - call < 60]
            
            # If at rate limit, wait
            if len(self.calls) >= self.calls_per_minute:
                await asyncio.sleep(1)
            
            # Record this call
            self.calls.append(now)

class MarketAnalysis:
    def __init__(self):
        self.news = newsapi
        self.alpha = alpha_vantage
        self.limiter = RateLimiter()

    async def get_market_trends(self, industry: str) -> Dict:
        try:
            await self.limiter.wait()
            
            # Get news articles
            news = self.news.get_everything(
                q=f"{industry} market trends",
                language='en',
                sort_by='relevancy',
                from_param=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                page_size=3
            )

            # Get market data
            try:
                market_data = self.alpha.get_sector()
            except Exception as e:
                market_data = None
                print(f"Alpha Vantage error: {e}")

            # Generate analysis using Gemini
            prompt = f"""Analyze {industry} market:
            1. Current market trends (3 points)
            2. Growth potential (2 metrics)
            3. Key challenges (2 points)
            Be specific and data-focused."""

            response = await model.generate_content(prompt)

            # Sample market trend data
            trend_data = [
                {"month": "Jan", "marketSize": 100, "growth": 15},
                {"month": "Feb", "marketSize": 120, "growth": 18},
                {"month": "Mar", "marketSize": 150, "growth": 22},
                {"month": "Apr", "marketSize": 180, "growth": 25}
            ]

            return {
                "news": news['articles'][:3],
                "market_data": market_data[0] if market_data else {},
                "analysis": response.text,
                "trends": trend_data
            }
        except Exception as e:
            print(f"Market analysis error: {str(e)}")
            return {
                "error": str(e),
                "trends": [
                    {"month": "Jan", "marketSize": 100, "growth": 15},
                    {"month": "Feb", "marketSize": 120, "growth": 18},
                    {"month": "Mar", "marketSize": 150, "growth": 22},
                    {"month": "Apr", "marketSize": 180, "growth": 25}
                ]
            }

class CustomerAnalysis:
    def __init__(self):
        self.model = model

    async def analyze_customers(self, startup: StartupInfo) -> Dict:
        try:
            prompt = f"""Create customer profile for {startup.name}:
            Industry: {startup.industry}
            Market: {startup.target_market}
            Demographics: {startup.target_demographics}

            Provide:
            1. Ideal customer persona
            2. Top 3 pain points
            3. Customer journey stages
            4. Buying behavior"""

            response = await self.model.generate_content(prompt)
            
            # Sample customer data
            customer_data = {
                "profiles": response.text,
                "demographics": [
                    {"group": "Age 25-34", "value": 40},
                    {"group": "Age 35-44", "value": 30},
                    {"group": "Age 45-54", "value": 20},
                    {"group": "Age 55+", "value": 10}
                ],
                "journey_stages": [
                    {"stage": "Awareness", "value": 100},
                    {"stage": "Consideration", "value": 70},
                    {"stage": "Decision", "value": 40},
                    {"stage": "Retention", "value": 25}
                ]
            }
            
            return customer_data
        except Exception as e:
            print(f"Customer analysis error: {str(e)}")
            return {
                "error": str(e),
                "demographics": [
                    {"group": "Age 25-34", "value": 40},
                    {"group": "Age 35-44", "value": 30},
                    {"group": "Age 45-54", "value": 20},
                    {"group": "Age 55+", "value": 10}
                ],
                "journey_stages": [
                    {"stage": "Awareness", "value": 100},
                    {"stage": "Consideration", "value": 70},
                    {"stage": "Decision", "value": 40},
                    {"stage": "Retention", "value": 25}
                ]
            }

class CompetitiveAnalysis:
    def __init__(self):
        self.bing_key = BING_API_KEY
        self.model = model

    async def analyze_competitors(self, startup: StartupInfo) -> Dict:
        try:
            competitors = startup.competitors[:3]
            
            # Get competitor data from Bing
            headers = {
                "Ocp-Apim-Subscription-Key": self.bing_key
            }
            
            competitor_data = []
            for competitor in competitors:
                try:
                    response = requests.get(
                        "https://api.bing.microsoft.com/v7.0/search",
                        headers=headers,
                        params={
                            "q": f"{competitor} company",
                            "count": 3,
                            "responseFilter": "Webpages"
                        }
                    )
                    if response.status_code == 200:
                        competitor_data.append(response.json())
                except Exception as e:
                    print(f"Bing API error for {competitor}: {e}")

            # Generate analysis
            prompt = f"""Compare {startup.name} with competitors ({', '.join(competitors) if competitors else 'similar companies'}):
            1. Competitive advantages
            2. Market positioning
            3. Feature comparison
            4. Growth opportunities"""

            response = await self.model.generate_content(prompt)

            # Sample comparison data
            comparison_data = {
                "analysis": response.text,
                "competitor_data": competitor_data,
                "comparison_metrics": [
                    {"aspect": "Features", "YourCompany": 85, "Competitor1": 90, "Competitor2": 75},
                    {"aspect": "Market Share", "YourCompany": 70, "Competitor1": 85, "Competitor2": 65},
                    {"aspect": "Innovation", "YourCompany": 95, "Competitor1": 80, "Competitor2": 70},
                    {"aspect": "Price", "YourCompany": 80, "Competitor1": 75, "Competitor2": 85}
                ]
            }
            
            return comparison_data
        except Exception as e:
            print(f"Competitive analysis error: {str(e)}")
            return {
                "error": str(e),
                "comparison_metrics": [
                    {"aspect": "Features", "YourCompany": 85, "Competitor1": 90, "Competitor2": 75},
                    {"aspect": "Market Share", "YourCompany": 70, "Competitor1": 85, "Competitor2": 65},
                    {"aspect": "Innovation", "YourCompany": 95, "Competitor1": 80, "Competitor2": 70},
                    {"aspect": "Price", "YourCompany": 80, "Competitor1": 75, "Competitor2": 85}
                ]
            }

def create_market_trends_chart(data):
    fig = go.Figure()
    
    # Market size trend
    fig.add_trace(go.Scatter(
        x=[d["month"] for d in data],
        y=[d["marketSize"] for d in data],
        name="Market Size",
        line=dict(color="#2563eb", width=3),
        mode='lines+markers'
    ))
    
    # Growth rate trend
    fig.add_trace(go.Scatter(
        x=[d["month"] for d in data],
        y=[d["growth"] for d in data],
        name="Growth Rate (%)", 
        line=dict(color="#16a34a", width=3, dash='dot'),
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Market Growth Trends",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        hovermode='x unified',
        yaxis=dict(title="Market Size (M$)", gridcolor='#f0f0f0'),
        yaxis2=dict(title="Growth Rate (%)", overlaying='y', side='right'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_customer_journey_funnel(data):
    fig = go.Figure(go.Funnel(
        y=[d["stage"] for d in data],
        x=[d["value"] for d in data],
        textinfo="value+percent initial",
        marker=dict(color=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd"])
    ))
    
    fig.update_layout(
        title="Customer Journey Funnel",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_competitor_analysis_chart(data):
    categories = [d["aspect"] for d in data]
    
    fig = go.Figure()
    
    # Your company
    fig.add_trace(go.Scatterpolar(
        r=[d["YourCompany"] for d in data],
        theta=categories,
        fill='toself',
        name='Your Company',
        line_color='#2563eb',
        opacity=0.8
    ))
    
    # Competitor 1
    fig.add_trace(go.Scatterpolar(
        r=[d["Competitor1"] for d in data],
        theta=categories,
        fill='toself',
        name='Main Competitor',
        line_color='#16a34a',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=False,
                gridcolor='#f0f0f0'
            )
        ),
        title="Competitive Analysis Radar",
        showlegend=True,
        height=450
    )
    
    return fig

def create_market_expansion_map(markets_data):
    # Create a simple choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=['USA', 'GBR', 'DEU', 'FRA', 'SGP', 'BRA'],
        z=[3, 2, 2, 2, 3, 1],
        text=['North America', 'UK', 'Germany', 'France', 'Singapore', 'Brazil'],
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title="Market Expansion Opportunities",
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        height=400
    )
    
    return fig

def landing_page():
    """Create an engaging landing page with startup information collection form."""
    st.title("AI Market Edge")
    st.write("Get instant market insights and strategic recommendations for your startup")

    with st.form("startup_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input(
                "Company Name*",
                help="Enter your company or startup name"
            )
            
            industry = st.selectbox(
                "Industry/Sector*",
                options=[
                    "Software & Technology",
                    "E-commerce & Retail",
                    "Healthcare & Biotech",
                    "Financial Services",
                    "Education & EdTech",
                    "Manufacturing",
                    "Professional Services",
                    "Media & Entertainment","Other"
                ],
                help="Select the primary industry your startup operates in"
            )
            
            target_market = st.multiselect(
                "Target Markets*",
                options=[
                    "North America",
                    "Europe",
                    "Asia Pacific",
                    "Latin America",
                    "Middle East",
                    "Africa"
                ],
                default=["North America"],
                help="Select all regions you plan to target"
            )

        with col2:
            demographics = st.text_input(
                "Target Customer Demographics",
                placeholder="e.g., B2B enterprises, young professionals 25-40",
                help="Describe your ideal customer profile"
            )
            
            description = st.text_area(
                "Business Description*",
                max_chars=500,
                height=100,
                help="Briefly describe your business idea and value proposition"
            )
        
        # Fix for empty label warnings
        st.subheader("Primary Business Goals*")
        goals = st.multiselect(
            "Select your main business objectives",  # Added label
            options=[
                "Market Research & Validation",
                "Customer Discovery",
                "Competitive Analysis",
                "Product Development",
                "Market Expansion",
                "Fundraising",
                "Partnership Development"
            ],
            help="Select your main business objectives"
        )
        
        st.subheader("Competitors")
        competitors = st.text_area(
            "List your main competitors",  # Added label
            placeholder="Enter competitor names (one per line)",
            height=100,
            help="List your main competitors (up to 5)"
        )

        col3, col4 = st.columns(2)
        with col3:
            stage = st.selectbox(
                "Company Stage",
                options=[
                    "Idea/Concept",
                    "MVP/Prototype",
                    "Early Revenue",
                    "Growth Stage",
                    "Scaling"
                ]
            )
        
        with col4:
            team_size = st.number_input(
                "Team Size",
                min_value=1,
                max_value=1000,
                value=1
            )

        submitted = st.form_submit_button("Generate Analysis")
        
        if submitted:
            if not all([company_name, industry, target_market, description, goals]):
                st.error("Please fill in all required fields marked with *")
                return None
            
            return StartupInfo(
                name=company_name,
                industry=industry,
                target_market=", ".join(target_market),
                target_demographics=demographics,
                description=description,
                goals=goals,
                competitors=[c.strip() for c in competitors.split('\n') if c.strip()][:5]
            )
    return None

async def run_analysis(startup: StartupInfo) -> Dict:
    """Run comprehensive market analysis using multiple data sources."""
    try:
        # Initialize analysis components
        market_analysis = MarketAnalysis()
        customer_analysis = CustomerAnalysis()
        competitive_analysis = CompetitiveAnalysis()
        
        # Progress indicator
        progress_text = "Analyzing market data..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Run analyses in parallel
        analyses = await asyncio.gather(
            market_analysis.get_market_trends(startup.industry),
            customer_analysis.analyze_customers(startup),
            competitive_analysis.analyze_competitors(startup),
            return_exceptions=True
        )
        
        # Update progress
        progress_bar.progress(50, text="Processing results...")
        
        # Process results
        results = {
            "market_analysis": analyses[0] if not isinstance(analyses[0], Exception) else {
                "error": str(analyses[0]),
                "trends": [
                    {"month": "Jan", "marketSize": 100, "growth": 15},
                    {"month": "Feb", "marketSize": 120, "growth": 18},
                    {"month": "Mar", "marketSize": 150, "growth": 22},
                    {"month": "Apr", "marketSize": 180, "growth": 25}
                ]
            },
            "customer_analysis": analyses[1] if not isinstance(analyses[1], Exception) else {
                "error": str(analyses[1]),
                "demographics": [
                    {"group": "25-34", "value": 40},
                    {"group": "35-44", "value": 30},
                    {"group": "45-54", "value": 20},
                    {"group": "55+", "value": 10}
                ],
                "journey_stages": [
                    {"stage": "Awareness", "value": 100},
                    {"stage": "Consideration", "value": 70},
                    {"stage": "Decision", "value": 40},
                    {"stage": "Retention", "value": 25}
                ]
            },
            "competitive_analysis": analyses[2] if not isinstance(analyses[2], Exception) else {
                "error": str(analyses[2]),
                "comparison_metrics": [
                    {"aspect": "Features", "YourCompany": 85, "Competitor1": 90, "Competitor2": 75},
                    {"aspect": "Market Share", "YourCompany": 70, "Competitor1": 85, "Competitor2": 65},
                    {"aspect": "Innovation", "YourCompany": 95, "Competitor1": 80, "Competitor2": 70},
                    {"aspect": "Price", "YourCompany": 80, "Competitor1": 75, "Competitor2": 85}
                ]
            }
        }
        
        # Final progress update
        progress_bar.progress(100, text="Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()
        
        return results
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        return {
            "market_analysis": {},
            "customer_analysis": {},
            "competitive_analysis": {}
        }

def display_dashboard(startup: StartupInfo, analysis_results: Dict):
    # Header with company name and logo placeholder
    st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 2rem;">
            <div style="width: 60px; height: 60px; background: #2563eb; border-radius: 12px; margin-right: 1rem;"></div>
            <div>
                <h1 style="margin: 0;">{startup.name}</h1>
                <p style="color: #6b7280; margin: 0;">{startup.industry} | {startup.target_market}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Add a "Start New Analysis" button at the top
    if st.button("Start New Analysis"):
        st.session_state.analysis_complete = False
        st.rerun()
    
    # Key Metrics Overview
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = {
        "Market Opportunity": {"value": "High", "color": "#2563eb", "icon": "📈"},
        "Customer Fit": {"value": "85%", "color": "#16a34a", "icon": "🎯"},
        "Competition Level": {"value": "Medium", "color": "#f59e0b", "icon": "⚡"},
        "Growth Potential": {"value": "Strong", "color": "#7c3aed", "icon": "🚀"}
    }
    
    for col, (metric, data) in zip([col1, col2, col3, col4], metrics.items()):
        col.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">{data['icon']} {metric}</div>
                <div class="metric-value" style="color: {data['color']}">{data['value']}</div>
            </div>
        """, unsafe_allow_html=True)

    # Main Dashboard Tabs
    tabs = st.tabs([
        "🔍 Market Analysis",
        "👥 Customer Discovery",
        "🎯 Competitive Intel",
        "📈 Product Evolution",
        "🌐 Market Expansion"
    ])
    
    # Market Analysis Tab
    with tabs[0]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        
        # Market Analysis Header
        st.subheader("Market Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            market_data = analysis_results.get("market_analysis", {})
            if "trends" in market_data:
                fig = create_market_trends_chart(market_data["trends"])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
                <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem;">
                    <h4>Key Insights</h4>
                    <ul style="margin: 0; padding-left: 1.2rem;">
                        <li>Market growing at 15% CAGR</li>
                        <li>Emerging technology adoption</li>
                        <li>Shift towards digital solutions</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            # Recent News Section
            st.markdown("<h4>Industry Updates</h4>", unsafe_allow_html=True)
            for article in market_data.get("news", [])[:2]:
                st.markdown(f"""
                    <div style="padding: 1rem; border-left: 4px solid #2563eb; 
                         background: #f8fafc; margin: 0.5rem 0; border-radius: 0 0.5rem 0.5rem 0;">
                        <h5 style="margin: 0;">{article['title']}</h5>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">{article['description']}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer Discovery Tab
    with tabs[1]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Customer Journey Analysis")
            customer_data = analysis_results.get("customer_analysis", {})
            if "journey_stages" in customer_data:
                fig = create_customer_journey_funnel(customer_data["journey_stages"])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Target Customer Profile")
            st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem;">
                    <h4 style="margin-top: 0;">Ideal Customer Persona</h4>
                    <ul style="margin: 0; padding-left: 1.2rem;">
                        <li><strong>Demographics:</strong> 25-45 years old professionals</li>
                        <li><strong>Pain Points:</strong> Time management, productivity</li>
                        <li><strong>Goals:</strong> Business growth, efficiency</li>
                        <li><strong>Buying Behavior:</strong> Research-driven, value-focused</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            # Demographics Distribution
            if "demographics" in customer_data:
                fig = px.pie(
                    customer_data["demographics"],
                    values="value",
                    names="group",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig.update_layout(title="Demographics Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Competitive Intelligence Tab
    with tabs[2]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        
        competitor_data = analysis_results.get("competitive_analysis", {})
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Competitive Landscape")
            if "comparison_metrics" in competitor_data:
                fig = create_competitor_analysis_chart(competitor_data["comparison_metrics"])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Competitive Advantages")
            st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem;">
                    <h4 style="margin-top: 0;">Your Edge</h4>
                    <ul>
                        <li>Innovative technology stack</li>
                        <li>Superior user experience</li>
                        <li>Competitive pricing model</li>
                        <li>Faster time to market</li>
                    </ul>
                    
                    <h4>Market Gaps</h4>
                    <ul>
                        <li>Underserved SMB segment</li>
                        <li>Mobile-first solutions</li>
                        <li>Integration capabilities</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Product Evolution Tab
    with tabs[3]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Feature Priority Matrix")
            
            # Create feature priority matrix
            features_df = pd.DataFrame({
                'Feature': ['Mobile App', 'API Integration', 'Analytics', 'Team Collab', 'Custom Reports'],
                'Priority': [90, 85, 80, 75, 70],
                'Effort': [80, 70, 60, 50, 40]
            })
            
            fig = px.scatter(
                features_df,
                x='Effort',
                y='Priority',
                text='Feature',
                size=[40] * len(features_df),
                color='Priority',
                color_continuous_scale='Blues'
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(
                title="Feature Priority vs. Effort",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Development Timeline")
            
            timeline_data = {
                "Q1 2024": ["Mobile app beta", "Initial API"],
                "Q2 2024": ["Analytics dashboard", "Team features"],
                "Q3 2024": ["Custom reporting", "Advanced API"],
                "Q4 2024": ["Mobile app v2", "Enterprise features"]
            }
            
            for quarter, milestones in timeline_data.items():
                st.markdown(f"""
                    <div style="padding: 1rem; border-left: 4px solid #2563eb; 
                         background: #f8fafc; margin: 0.5rem 0; border-radius: 0 0.5rem 0.5rem 0;">
                        <h4 style="margin: 0;">{quarter}</h4>
                        <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
                            {''.join(f'<li>{m}</li>' for m in milestones)}
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Market Expansion Tab
    with tabs[4]:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        
        st.subheader("Global Expansion Opportunities")
        
        # World map showing expansion opportunities
        fig = create_market_expansion_map({})
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem;">
                    <h4 style="margin-top: 0;">Priority Markets</h4>
                    <ul>
                        <li><strong>Primary:</strong> North America, Europe</li>
                        <li><strong>Secondary:</strong> Asia Pacific, Latin America</li>
                    </ul>
                    
                    <h4>Market Requirements</h4>
                    <ul>
                        <li>Local partnerships</li>
                        <li>Regulatory compliance</li>
                        <li>Cultural adaptation</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style="background: #f8fafc; padding: 1.5rem; border-radius: 0.5rem;">
                    <h4 style="margin-top: 0;">Expansion Readiness</h4>
                    <ul>
                        <li><strong>Technical:</strong> 80% Ready</li>
                        <li><strong>Operational:</strong> 70% Ready</li>
                        <li><strong>Marketing:</strong> 65% Ready</li>
                        <li><strong>Support:</strong> 75% Ready</li>
                    </ul>
                    
                    <h4>Next Steps</h4>
                    <ol>
                        <li>Market validation research</li>
                        <li>Local partner identification</li>
                        <li>Compliance assessment</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Update the main function to use the new dashboard layout
def main():
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    
    if not st.session_state.analysis_complete:
        startup_info = landing_page()
        if startup_info:
            with st.spinner("Generating comprehensive market analysis..."):
                analysis_results = asyncio.run(run_analysis(startup_info))
                st.session_state.startup_info = startup_info
                st.session_state.analysis_results = analysis_results
                st.session_state.analysis_complete = True
                st.success("Analysis complete! Displaying insights...")
                st.rerun()
    else:
        display_dashboard(
            st.session_state.startup_info,
            st.session_state.analysis_results
        )

if __name__ == "__main__":
    main()
