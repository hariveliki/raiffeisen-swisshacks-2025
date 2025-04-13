import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.agents.base_agent import BaseAgent
from app.utils.data_loader import DataLoader
from app.utils.vector_store import VectorStore


class ProductPortfolioCheckerAgent(BaseAgent):
    """Agent responsible for checking product inquiries against the existing portfolio."""

    def __init__(self, *args, **kwargs):
        """Initialize the product portfolio checker agent."""
        super().__init__(*args, **kwargs)

    def load_data(self):
        """Load transcript and product portfolio data."""
        print("Loading transcript and product portfolio...")
        self.transcript = DataLoader.load_transcript()
        self.product_portfolio = DataLoader.load_product_portfolio()
        self.product_data = DataLoader.load_product_portfolio()
        vector_store = VectorStore()
        self.product_vector_store = vector_store.create_or_load(
            self.product_data, "product_portfolio"
        )

    def extract_product_inquiries(self) -> dict:
        """
        Extract product-related inquiries from the transcript.
        Returns a dictionary of product inquiries.
        """
        prompt_template = """
        Extract ONLY product-related inquiries or requests from this conversation.
        Focus on specific products or services the client is asking about.
        DO NOT include general financial advice requests.
        
        Conversation:
        {transcript}
        
        Return a JSON object with these fields (only include fields that are explicitly mentioned):
        {{
            "product_inquiries": [
                {{
                    "product_type": "type of product/service inquired about",
                    "specific_need": "specific need or requirement mentioned",
                    "context": "brief context of the inquiry"
                }}
            ]
        }}
        
        If no product inquiries are mentioned, return an empty list for product_inquiries.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a product portfolio expert designed to output JSON. Extract only explicit product inquiries. Do not make assumptions or general financial advice requests.",
            },
            {
                "role": "user",
                "content": prompt_template.format(transcript=self.transcript),
            },
        ]

        response = self.get_completion(
            messages, temperature=0, response_format={"type": "json_object"}
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error: Could not parse product inquiries as JSON")
            return {"product_inquiries": []}

    def check_against_portfolio(self, inquiries: dict) -> list:
        """
        Check product inquiries against the existing portfolio.
        Returns a list of findings about product availability.
        """
        findings = []

        for inquiry in inquiries.get("product_inquiries", []):
            product_type = inquiry.get("product_type", "")
            specific_need = inquiry.get("specific_need", "")

            search_query = f"{product_type} {specific_need}"
            print(f"SEARCH QUERY: {search_query}")
            search_results = self.product_vector_store.similarity_search(
                search_query, k=1
            )
            print(f"SEARCH RESULTS: {search_results}")

            if search_results:
                relevance_prompt = f"""
                Given the following product inquiry and retrieved result, determine if the result is relevant to the inquiry.
                Return only "true" or "false".

                Inquiry: {search_query}
                Retrieved Result: {search_results[0].page_content}
                """

                relevance_messages = [
                    {
                        "role": "system",
                        "content": "You are a relevance checker. Respond only with 'true' or 'false'.",
                    },
                    {"role": "user", "content": relevance_prompt},
                ]

                relevance_response = (
                    self.get_completion(relevance_messages, temperature=0)
                    .strip()
                    .lower()
                )

                if relevance_response != "true":
                    finding = f"- During the meeting, Haris asked whether Raiffeisen has a solution for {product_type.lower()} which does not currently exist."
                    findings.append(finding)
                    continue

            if not search_results or search_results[0].page_content.strip() == "":
                finding = f"- During the meeting, Haris asked whether Raiffeisen has a solution for {product_type.lower()} which does not currently exist."
                findings.append(finding)

        return findings

    def generate_portfolio_report(self):
        """
        Generate a report of product inquiries and portfolio gaps.
        Returns a list of findings in a consistent format.
        """
        inquiries = self.extract_product_inquiries()
        findings = self.check_against_portfolio(inquiries)
        return findings

    def run(self):
        """
        Run the product portfolio checker process.

        Returns:
            list: List of findings about product availability and gaps.
        """
        self.load_data()
        print("Generating product portfolio report...")
        findings = self.generate_portfolio_report()
        return findings


if __name__ == "__main__":
    portfolio_agent = ProductPortfolioCheckerAgent()
    findings = portfolio_agent.run()
    with open("output/product_portfolio_findings.txt", "w") as f:
        f.write("\n".join(findings))
