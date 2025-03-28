import asyncio
import os
import requests
import json
import sys
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
from together import Together
from bs4 import BeautifulSoup

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Check if API keys are available
if not TOGETHER_API_KEY:
    print("Error: TOGETHER_API_KEY not found in .env file")
    sys.exit(1)
if not SERPER_API_KEY:
    print("Error: SERPER_API_KEY not found in .env file")
    sys.exit(1)

# Initialize Together AI client
client = Together(api_key=TOGETHER_API_KEY)

# Define data models
class SourceType(BaseModel):
    title: str
    link: str

class SerperResponse(BaseModel):
    organic: List[SourceType]

async def fetch_and_parse(source: Dict[str, str]) -> Dict[str, str]:
    """
    Fetch content from a URL and extract the text.
    
    Args:
        source: Dictionary containing the URL to fetch
        
    Returns:
        Dictionary with the original source data plus extracted content
    """
    try:
        # Fetch webpage with 5 second timeout
        response = requests.get(source["url"], timeout=5)
        response.raise_for_status()
        
        # Parse HTML and extract text
        doc = BeautifulSoup(response.text, "html.parser")
        parsed_content = doc.get_text(strip=True)
        
        # Limit content length to 20000 characters
        cleaned_content = parsed_content[:20000]
        
        return {**source, "fullContent": cleaned_content}
    except Exception as e:
        print(f"Error fetching {source['url']}: {str(e)}")
        return {**source, "fullContent": "not available"}


async def get_search_results(question: str) -> List[Dict[str, str]]:
    """
    Search the web for relevant sources using Google Serper API.
    
    Args:
        question: The user's question to search for
        
    Returns:
        List of sources with title and URL
    """
    try:
        # Send search request to Serper API
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            data=json.dumps({"q": question, "num": 6}),
        )
        response.raise_for_status()
        
        # Parse response and extract sources
        raw_json = response.json()
        data = SerperResponse.model_validate(raw_json)
        
        # Format results
        results = [{"name": result.title, "url": result.link} for result in data.organic]
        
        # Print sources
        print("===== Sources ======")
        for i, result in enumerate(results):
            print(f"Source {i + 1}: {result['name']} [{result['url']}]")
        
        return results
    except Exception:
        return []


async def generate_answer(question: str, sources: List[Dict[str, str]]) -> str:
    """
    Generate an answer to the user's question based on the sources.
    
    Args:
        question: The user's question
        sources: List of sources to use for answering
        
    Returns:
        Generated answer text
    """
    # Check if we have any sources
    if not sources:
        return "Unable to answer this question because no sources were found."
    
    # Fetch and parse all sources in parallel
    final_results = await asyncio.gather(*map(fetch_and_parse, sources))
    
    # Create prompt for the AI model
    main_answer_prompt = f"""
    Given a user question and some context, please write a clean, concise and accurate answer to the question based on the context. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context when crafting your answer.

    Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

    Here are the set of contexts:
    {"".join([f"[[citation:{index}]] {result['fullContent']} " for index, result in enumerate(final_results)])}

    Remember, don't blindly repeat the contexts verbatim and don't tell the user how you used the citations – just respond with the answer. It is very important for my career that you follow these instructions. Here is the user question:
    """

    try:
        # Call the AI model to generate an answer
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": main_answer_prompt},
                {"role": "user", "content": question},
            ],
        )
        
        # Extract and print the answer
        print("===== Answer ======\n")
        answer = response.choices[0].message.content.strip()
        print(answer)
        
        return answer
    except Exception:
        return "An error occurred while generating the answer."


async def get_similar_questions(question: str) -> str:
    """
    Generate related questions based on the user's original question.
    
    Args:
        question: The user's original question
        
    Returns:
        JSON string containing three related questions
    """
    # Define prompt for the AI model
    prompt = """
    You are a helpful assistant that helps the user to ask related questions, based on user's original question. Please identify worthwhile topics that can be follow-ups, and write 3 questions no longer than 20 words each. Please make sure that specifics, like events, names, locations, are included in follow up questions so they can be asked standalone. For example, if the original question asks about "the Manhattan project", in the follow up question, do not just say "the project", but use the full name "the Manhattan project". Your related questions must be in the same language as the original question. Please provide these 3 related questions as a JSON array of 3 strings. Do NOT repeat the original question. ONLY return the JSON array, I will get fired if you don't return JSON. Here is the user's original question:
    """
    
    try:
        # Call the AI model to generate related questions
        response = client.chat.completions.create(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
        )
        
        # Print the related questions
        print("===== Similar Questions ======")
        content = response.choices[0].message.content
        print(content)
        
        # Validate JSON format
        try:
            json.loads(content)
            return content
        except json.JSONDecodeError:
            return json.dumps(["Similar question 1?", "Similar question 2?", "Similar question 3?"])
    except Exception:
        return json.dumps(["Similar question 1?", "Similar question 2?", "Similar question 3?"])


async def main():
    """
    Main function that handles the entire process.
    """
    # Get question from command line args or use default
    question = sys.argv[1] if len(sys.argv) > 1 else "What are some fun things to do in San Francisco?"
    
    # Get search results
    sources = await get_search_results(question)
    
    # Generate answer and similar questions in parallel
    await asyncio.gather(generate_answer(question, sources), get_similar_questions(question))


# Run the main function
asyncio.run(main())
