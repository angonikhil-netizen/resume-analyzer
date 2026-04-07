
import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import PyPDF2


st.set_page_config(page_title="AI Resume Intelligence", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
    }
    p, .stmarkdown, .stwrite, .stfileuploader{
            color: white !important;
            } 

    a {
        color: #00f2fe !important; /* Bright Cyan/Blue */
        text-decoration: none;
        font-weight: bold;
        
    }

    a:hover {
        color: #ffffff !important;
        text-shadow: 0px 0px 15px rgba(255, 255, 255, 0.9);
    }       
    .stButton>button {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        color: black;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        height: 3em;
    }
    .report-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
            
    
    [data-testid="stFileUploader"] button {
        background-color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        
        display: flex !important;
        justify-content: flex-start !important; 
        align-items: center !important;
        gap: 0px !important
        
        padding: 5px 10px !important; 
        width: auto !important; 
    }

    
    [data-testid="stFileUploader"] button::after {
        content: "upload file"; 
        color: grey !important;
        font-size: 16px;
            gap: 0px !important;
        
        margin-left: 5px !important; 
    }



    
    [data-testid="stFileUploader"] small {
        color: white !important;
        opacity: 0.8;
            gap: 0px !important;
            margin: 0 !important;
    }

    
            

   

     [data-testid="stFileUploader"] button {
        background-color: white !important;
        color: grey !important;
    }
            
    [data-testid="stFileUploader"] button::after {
        content: " upload file"; 
        color: grey !important;
        font-size: 16px;
        
        margin-left: 5px !important; 
    }
    
    
       
    </style>
    """, unsafe_allow_html=True)



load_dotenv()
hf_token = os.getenv('HF_TOKEN')

def get_analysis(text):
    client = InferenceClient(token=hf_token)
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    messages = [
        {"role": "system", "content": "You are a professional Resume Expert."},
        {"role": "user", "content": f"Analyze this resume. Give Summary, Skills, Strengths, Weaknesses, and Recommendation:\n\n{text}"}
    ]
    response = client.chat.completions.create(model=model_id, messages=messages, max_tokens=1000)
    return response.choices[0].message.content


st.title("📑 Professional AI Resume Analyzer")
st.write("Upload a PDF to receive an instant AI career evaluation.")
st.markdown("Built by Nikhil.")
st.markdown("Contact: [angonikhil@gmail.com](mailto:angonikhil@gmail.com?subject=Resume%20Analysis%20Inquiry)")


st.markdown("### Upload PDF:") 
file = st.file_uploader("", type="pdf",label_visibility="collapsed")
if file and st.button("START ANALYSIS"):
    with st.spinner("Analyzing..."):
        reader = PyPDF2.PdfReader(file)
        full_text = "".join([p.extract_text() for p in reader.pages])
        
        result = get_analysis(full_text)
        st.markdown("### Analysis Report")
        st.markdown(f'<div class="report-box">{result}</div>', unsafe_allow_html=True)