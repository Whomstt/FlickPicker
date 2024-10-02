from django.shortcuts import render
from django.http import JsonResponse
import re
import requests
from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

def chatbot_form(request):
    response = None
    if request.method == "POST":
        db = SQLDatabase.from_uri("postgresql://root:root@db:5432/db")
        db_description = (
            "The database consists of a table 'Film'."
            "The 'Film' table contains details on films"
            "It contains the following collumns: id, director, genre, release_date, title"
        )
        prompt = request.POST.get("prompt")
        full_prompt = db_description + " " + prompt
        ollama_url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "llama3.1",
            "prompt": full_prompt
        }
        headers = {
            "Content-Type": "application/json"
        }
        try:
            ollama_response = requests.post(ollama_url, json = payload, headers = headers)
            ollama_response.raise_for_status()
            ollama_data = ollama_response.json()
            generated_text = "".join(part["response"] for part in ollama_data if "response" in part)
        except requests.RequestException as e:
            generated_text = f"Error: {str(e)}"

        response = generated_text

        def extract_sql_query(response):
            pattern = re.compile(r"SQLQuery:\s*(.*)")
            match = pattern.search(response)
            return match.group(1).strip() if match else None
        
        sql_query = extract_sql_query(response)

        if sql_query:
            result = db.run(sql_query)
            response = f"Query: {sql_query}, Result: {result}"
    
    return render(request, "chatbot_form.html", {"response": response})