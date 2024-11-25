from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import pandas as pd
import os
from dotenv import load_dotenv
import asyncio
from datetime import datetime, timedelta
import json
import numpy as np

# Load environment variables
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")

# Initialize Groq model
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama2-70b-4096",
    temperature=0.7,
    max_tokens=4096
)

# Define data structures
@dataclass
class StartupInfo:
    name: str
    industry: str
    target_market: str
    target_demographics: str
    description: str
    goals: List[str]
    competitors: List[str]

class MarketAnalysisInput(BaseModel):
    industry: str = Field(description="Industry to analyze")
    timeframe: str = Field(description="Analysis timeframe (e.g., '1 year')")
    region: str = Field(description="Geographic region for analysis")

class CompetitorAnalysisInput(BaseModel):
    company_name: str = Field(description="Company to analyze")
    competitors: List[str] = Field(description="List of competitors")
    industry: str = Field(description="Industry sector")

class CustomerAnalysisInput(BaseModel):
    target_market: str = Field(description="Target market description")
    demographics: str = Field(description="Target demographics")
    industry: str = Field(description="Industry sector")

# Initialize tools
search = DuckDuckGoSearchRun()
news = YahooFinanceNewsTool()
alpha_vantage = AlphaVantageAPIWrapper(alpha_vantage_api_key=ALPHA_VANTAGE_KEY)

def parse_llm_response(text: str) -> Dict:
    """Helper function to parse structured data from LLM response"""
    try:
        # Try to find JSON-like structure
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            data_str = text[start:end]
            return json.loads(data_str)
    except:
        pass
    return {"analysis": text}

# Custom tools
def analyze_market(input_data: MarketAnalysisInput) -> str:
    """Analyzes market trends and opportunities for a given industry"""
    search_results = search.run(f"{input_data.industry} market trends {input_data.region} last {input_data.timeframe}")
    news_results = news.run(f"{input_data.industry} market news")
    
    prompt = f"""
    Analyze the following market data and provide insights in JSON format:
    Search Results: {search_results}
    News: {news_results}
    
    Return a JSON object with the following structure:
    {{
        "market_trends": [list of 3 key trends],
        "growth_opportunities": [list of 2-3 opportunities],
        "challenges": [list of 2-3 challenges],
        "market_size": {{
            "current": "estimated current market size",
            "growth_rate": "estimated growth rate"
        }},
        "analysis_summary": "detailed analysis in 2-3 paragraphs"
    }}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def analyze_competitors(input_data: CompetitorAnalysisInput) -> str:
    """Analyzes competitive landscape and positioning"""
    competitors_data = []
    for competitor in input_data.competitors:
        search_result = search.run(f"{competitor} company {input_data.industry} analysis revenue products")
        competitors_data.append(f"{competitor}: {search_result}")
    
    prompt = f"""
    Analyze competitive landscape for {input_data.company_name} in JSON format:
    Industry: {input_data.industry}
    Competitors Data: {competitors_data}
    
    Return a JSON object with the following structure:
    {{
        "competitive_advantages": [list of advantages],
        "competitive_disadvantages": [list of disadvantages],
        "market_positioning": "detailed positioning analysis",
        "competitor_comparison": {{
            "strengths": [list of strengths],
            "weaknesses": [list of weaknesses]
        }},
        "recommendations": [list of strategic recommendations]
    }}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def analyze_customers(input_data: CustomerAnalysisInput) -> str:
    """Analyzes target customers and creates customer personas"""
    prompt = f"""
    Create detailed customer analysis for {input_data.industry} in JSON format:
    Target Market: {input_data.target_market}
    Demographics: {input_data.demographics}
    
    Return a JSON object with the following structure:
    {{
        "personas": [
            {{
                "name": "persona name",
                "description": "detailed description",
                "pain_points": [list of pain points],
                "goals": [list of goals]
            }}
        ],
        "buying_behavior": {{
            "triggers": [list of purchase triggers],
            "decision_factors": [list of decision factors],
            "barriers": [list of purchase barriers]
        }},
        "customer_journey": [
            {{
                "stage": "stage name",
                "description": "stage description",
                "touchpoints": [list of touchpoints]
            }}
        ],
        "recommendations": [list of recommendations]
    }}
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Create tools list
tools = [
    Tool(
        name="Market Analysis",
        func=analyze_market,
        description="Analyzes market trends and opportunities for a given industry"
    ),
    Tool(
        name="Competitor Analysis",
        func=analyze_competitors,
        description="Analyzes competitive landscape and positioning"
    ),
    Tool(
        name="Customer Analysis",
        func=analyze_customers,
        description="Analyzes target customers and creates customer personas"
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Searches the internet for information"
    ),
    Tool(
        name="News",
        func=news.run,
        description="Gets latest news articles"
    )
]

# Create agent prompt
system_prompt = """You are an expert business and market analysis AI agent.
Your goal is to help startups understand their market, competitors, and customers.
Follow a systematic approach:
1. Gather and analyze market data
2. Study competitors
3. Understand target customers
4. Provide actionable insights
Always provide structured data in your responses and explain your reasoning."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Visualization functions
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

def create_competitor_radar_chart(data):
    categories = list(data[0].keys())[1:]  # Skip the 'aspect' key
    fig = go.Figure()

    for competitor in data[0].keys()[1:]:
        values = [d[competitor] for d in data]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=competitor
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True
    )
    return fig

def create_customer_journey_funnel(data):
    fig = go.Figure(go.Funnel(
        y=[d["stage"] for d in data],
        x=[d["value"] for d in data],
        textinfo="value+percent initial"
    ))
    
    fig.update_layout(
        title="Customer Journey Funnel",
        showlegend=False
    )
    
    return fig

# Streamlit interface
def main():
    st.set_page_config(
        layout="wide",
        page_title="AI Market Edge",
        page_icon="📊",
        initial_sidebar_state="expanded"
    )

    # CSS styling
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
        .insight-card {
            background: white;
            padding: 1.5rem;
            border-radius: 0.75rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("AI Market Edge")
    st.write("Powered by LangChain + Groq LLama")

    # Startup information form
    with st.form("startup_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company_name = st.text_input("Company Name*")
            industry = st.selectbox(
                "Industry*",
                ["Software & Technology", "E-commerce", "Healthcare", "Financial Services", 
                 "Manufacturing", "Education", "Real Estate", "Consumer Goods"]
            )
            target_market = st.multiselect(
                "Target Markets*",
                ["North America", "Europe", "Asia Pacific", "Latin America", 
                 "Middle East", "Africa"]
            )

        with col2:
            demographics = st.text_input("Target Demographics", 
                                       placeholder="e.g., B2B enterprises, professionals 25-45")
            description = st.text_area("Business Description*")
            competitors = st.text_area("Main Competitors (one per line)")

        goals = st.multiselect(
            "Business Goals*",
            ["Market Entry", "Growth", "Product Development", "Customer Acquisition",
             "Market Expansion", "Competitive Positioning", "Brand Building"]
        )

        submitted = st.form_submit_button("Analyze")

        if submitted:
            if not all([company_name, industry, target_market, description, goals]):
                st.error("Please fill in all required fields")
                return

            startup_info = StartupInfo(
                name=company_name,
                industry=industry,
                target_market=", ".join(target_market),
                target_demographics=demographics,
                description=description,
                goals=goals,
                competitors=[c.strip() for c in competitors.split('\n') if c.strip()]
            )

            # Run AI analysis
            with st.spinner("Analyzing your business..."):
                try:
                    # Market analysis
                    market_input = MarketAnalysisInput(
                        industry=startup_info.industry,
                        timeframe="1 year",
                        region=", ".join(target_market)
                    )
                    market_analysis = agent_executor.invoke({
                        "input": f"Analyze the market for {startup_info.industry} industry in {', '.join(target_market)}"
                    })
                    market_data = parse_llm_response(market_analysis["output"])

                    # Competitor analysis
                    competitor_input = CompetitorAnalysisInput(
                        company_name=startup_info.name,
                        competitors=startup_info.competitors,
                        industry=startup_info.industry
                    )
                    competitor_analysis = agent_executor.invoke({
                        "input": f"Analyze competitors for {startup_info.name} in {startup_info.industry}"
                    })
                    competitor_data = parse_llm_response(competitor_analysis["output"])

                    # Customer analysis
                    customer_input = CustomerAnalysisInput(
                        target_market=startup_info.target_market,
                        demographics=startup_info.target_demographics,
                        industry=startup_info.industry
                    )
                    customer_analysis = agent_executor.invoke({
                        "input": f"Analyze target customers for {startup_info.name} in {startup_info.industry}"
                    })
                    customer_data = parse_llm_response(customer_analysis["output"])

                    # Display results in tabs
                    tabs = st.tabs(["Market Analysis", "Competitor Analysis", "Customer Analysis"])

                    with tabs[0]:
                        st.markdown("### Market Analysis")
                        
                        # Market metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Market Size", 
                                     market_data.get("market_size", {}).get("current", "N/A"))
                        with col2:
                            st.metric("Growth Rate", 
                                     market_data.get("market_size", {}).get("growth_rate", "N/A"))
                        with col3:
                            st.metric("Market Sentiment", "Positive")

                        # Market trends visualization
                        st.markdown("#### Market Trends")
                        # Sample data for visualization
                        trend_data = [
                            {"month": "Jan", "marketSize": 100, "growth": 15},
                            {"month": "Feb", "marketSize": 120, "growth": 18},
                            {"month": "Mar", "marketSize": 150, "growth": 22},
                            {"month": "Apr", "marketSize": 180, "growth": 25}
                        ]
                        fig = create_market_trends_chart(trend_data)
                        st.plotly_chart(fig, use_container_width=True)

                        # Market insights
                        st.markdown("#### Key Insights")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("##### Market Trends")
                            for trend in market_data.get("market_trends", []):
                                st.markdown(f"- {trend}")

                            st.markdown("##### Growth Opportunities")
                            for opportunity in market_data.get("growth_opportunities", []):
                                st.markdown(f"- {opportunity}")

                        with col2:
                            st.markdown("##### Challenges")
                            for challenge in market_data.get("challenges", []):
                                st.markdown(f"- {challenge}")

                        # Detailed analysis
                        with st.expander("View Detailed Analysis"):
                            st.markdown(market_data.get("analysis_summary", ""))

                    with tabs[1]:
                        st.markdown("### Competitor Analysis")
                        
                        # Competitive positioning
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Competitive Advantages")
                            for advantage in competitor_data.get("competitive_advantages", []):
                                st.markdown(f"- {advantage}")
                            
                            st.markdown("#### Market Positioning")
                            st.markdown(competitor_data.get("market_positioning", ""))

                        with col2:
                            # Competitor comparison visualization
                            comparison_data = [
                                {"aspect": "Features", "YourCompany": 85, "Competitor1": 90, "Competitor2": 75},
                                {"aspect": "Market Share", "YourCompany": 70, "Competitor1": 85, "Competitor2": 65},
                                {"aspect": "Innovation", "YourCompany": 95, "Competitor1": 80, "Competitor2": 70},
                                {"aspect": "Price", "YourCompany": 80, "Competitor1": 75, "Competitor2": 85}
                            ]
                            fig = create_competitor_radar_chart(comparison_data)
                            st.plotly_chart(fig, use_container_width=True)

                        # Strategic recommendations
                        st.markdown("#### Strategic Recommendations")
                        for rec in competitor_data.get("recommendations", []):
                            st.markdown(f"- {rec}")

                        # Detailed competitor analysis
                        with st.expander("View Detailed Competitor Analysis"):
                            st.markdown("##### Strengths")
                            for strength in competitor_data.get("competitor_comparison", {}).get("strengths", []):
                                st.markdown(f"- {strength}")
                            
                            st.markdown("##### Weaknesses")
                            for weakness in competitor_data.get("competitor_comparison", {}).get("weaknesses", []):
                                st.markdown(f"- {weakness}")

                    with tabs[2]:
                        st.markdown("### Customer Analysis")
                        
                        # Customer personas
                        st.markdown("#### Customer Personas")
                        for persona in customer_data.get("personas", []):
                            with st.expander(f"Persona: {persona.get('name', 'Unknown')}"):
                                st.markdown(persona.get("description", ""))
                                
                                st.markdown("##### Pain Points")
                                for point in persona.get("pain_points", []):
                                    st.markdown(f"- {point}")
                                
                                st.markdown("##### Goals")
                                for goal in persona.get("goals", []):
                                    st.markdown(f"- {goal}")

                        # Buying behavior
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Buying Behavior")
                            behavior = customer_data.get("buying_behavior", {})
                            
                            st.markdown("##### Purchase Triggers")
                            for trigger in behavior.get("triggers", []):
                                st.markdown(f"- {trigger}")
                            
                            st.markdown("##### Decision Factors")
                            for factor in behavior.get("decision_factors", []):
                                st.markdown(f"- {factor}")

                        with col2:
                            # Customer journey visualization
                            journey_data = [
                                {"stage": "Awareness", "value": 100},
                                {"stage": "Consideration", "value": 70},
                                {"stage": "Decision", "value": 40},
                                {"stage": "Retention", "value": 25}
                            ]
                            fig = create_customer_journey_funnel(journey_data)
                            st.plotly_chart(fig, use_container_width=True)

                        # Customer journey details
                        st.markdown("#### Customer Journey Analysis")
                        for stage in customer_data.get("customer_journey", []):
                            with st.expander(f"Stage: {stage.get('stage', 'Unknown')}"):
                                st.markdown(stage.get("description", ""))
                                st.markdown("##### Key Touchpoints")
                                for touchpoint in stage.get("touchpoints", []):
                                    st.markdown(f"- {touchpoint}")

                        # Recommendations
                        st.markdown("#### Customer Strategy Recommendations")
                        for rec in customer_data.get("recommendations", []):
                            st.markdown(f"- {rec}")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")
                    st.warning("Some visualizations may show sample data due to the error.")

if __name__ == "__main__":
    main()