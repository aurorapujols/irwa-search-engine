import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env


class RAGGenerator:

    PROMPT_TEMPLATE = """
        You are an expert product advisor for an e-commerce system. You will receive retrieved products with structured metadata.

        1. Evaluate if any retrieved product meaningfully satisfies the user's request:
        - A product satisfies the request ONLY if its attributes (price, features, category, specs, etc.) explicitly support the request.
        - Cite only fields present in the retrieved products as evidence.
        - If none qualify, respond ONLY AND EXACTLY: "There are no good products that fit the request based on the retrieved results."
        - If the user query is empty or nonsensical, respond ONLY AND EXACTLY: "The user request is empty or does not make sense."
        - Do NOT infer or guess missing information.

        2. Present the recommendation clearly in this format using plain text and maximum 150 words:
        - Best Product: Title  
        - Why: Explanation in plain language why it is the best fit, referring only retrieved product fields if needed  
        - Alternative (optional): ONE alternative, with its attributes  

        IMPORTANT: Do not mention anything not in the retrieved metadata.

        ## Retrieved Products:
        {retrieved_results}

        ## User Request:
        {user_query}

    """ 

    def generate_response(
        self, user_query: str, retrieved_results: list, top_N: int = 20
    ) -> dict:
        """
        Generate a response using the retrieved search results.
        Returns:
            dict: Contains the generated suggestion and the quality evaluation.
        """
        DEFAULT_ANSWER = "RAG is not available. Check your credentials (.env file) or account limits."
        try:
            client = Groq(
                api_key=os.environ.get("GROQ_API_KEY"),
            )
            model_name = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

            # Format the retrieved results for the prompt
            formatted_results = "\n".join([ f"""- PID: {res.pid}, Title: {res.title}, Price: {res.selling_price}, 
                                           Rating: {res.average_rating}, Category: {res.category}, Brand: {res.brand}, 
                                           Description: {res.description}, Features: {res.product_details}"""
                                        for res in retrieved_results[:top_N]
                                    ])

            prompt = self.PROMPT_TEMPLATE.format(
                retrieved_results=formatted_results, user_query=user_query
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
            )

            generation = chat_completion.choices[0].message.content
            return generation
        except Exception as e:
            print(f"Error during RAG generation: {e}")
            return DEFAULT_ANSWER
