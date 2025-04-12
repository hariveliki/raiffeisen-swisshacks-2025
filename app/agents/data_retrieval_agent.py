from app.agents.base_agent import BaseAgent
from app.utils.vector_store import VectorStore
from app.utils.data_loader import DataLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class DataRetrievalAgent(BaseAgent):
    """Agent responsible for retrieving relevant data from client state and product portfolio."""

    def __init__(self, *args, **kwargs):
        """Initialize the data retrieval agent."""
        super().__init__(*args, **kwargs)
        self.client_data = None
        self.product_data = None
        self.client_vector_store = None
        self.product_vector_store = None

    def load_data(self):
        """Load and index client and product data."""
        # Load raw data
        print("Loading client state data...")
        self.client_data = DataLoader.load_client_state()

        print("Loading product portfolio data...")
        self.product_data = DataLoader.load_product_portfolio()

        # Create vector stores for semantic search
        print("Creating vector stores for semantic search...")
        vector_store = VectorStore()

        # For client data, convert dictionary to text format for indexing
        if self.client_data:
            client_text = "\n".join([f"{k}: {v}" for k, v in self.client_data.items()])
            self.client_vector_store = vector_store.create_or_load(
                client_text, "client_state"
            )

        # For product data, directly index the text
        if self.product_data:
            self.product_vector_store = vector_store.create_or_load(
                self.product_data, "product_portfolio"
            )

        print("Data loading and indexing complete.")

    def retrieve_client_info(self, query, k=3):
        """
        Retrieve relevant client information based on a query.

        Args:
            query (str): The query to search for in the client data.
            k (int): Number of results to return.

        Returns:
            dict: Dictionary with retrieved information and raw results.
        """
        if not self.client_vector_store:
            print("Client vector store not initialized. Loading data...")
            self.load_data()

        # Perform semantic search
        results = []
        if self.client_vector_store:
            vector_store = VectorStore()
            vector_store.vector_store = self.client_vector_store
            results = vector_store.search(query, k=k)

        # Process results using LLM to format a response
        if results:
            prompt_template = """
            Based on the following client information, provide a concise summary relevant to: {query}
            
            Client Information:
            {results}
            
            Summary:
            """

            prompt = PromptTemplate(
                input_variables=["query", "results"], template=prompt_template
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Format results into a single string
            results_text = "\n".join([doc.page_content for doc in results])

            # Get summary from LLM
            response = chain.run(query=query, results=results_text)

            return {"summary": response, "raw_results": results}

        return {"summary": "No relevant client information found.", "raw_results": []}

    def retrieve_product_info(self, query, k=3):
        """
        Retrieve relevant product information based on a query.

        Args:
            query (str): The query to search for in the product data.
            k (int): Number of results to return.

        Returns:
            dict: Dictionary with retrieved information and raw results.
        """
        if not self.product_vector_store:
            print("Product vector store not initialized. Loading data...")
            self.load_data()

        # Perform semantic search
        results = []
        if self.product_vector_store:
            vector_store = VectorStore()
            vector_store.vector_store = self.product_vector_store
            results = vector_store.search(query, k=k)

        # Process results using LLM to format a response
        if results:
            prompt_template = """
            Based on the following product information, provide a concise summary relevant to: {query}
            
            Product Information:
            {results}
            
            Summary:
            """

            prompt = PromptTemplate(
                input_variables=["query", "results"], template=prompt_template
            )

            chain = LLMChain(llm=self.llm, prompt=prompt)

            # Format results into a single string
            results_text = "\n".join([doc.page_content for doc in results])

            # Get summary from LLM
            response = chain.run(query=query, results=results_text)

            return {"summary": response, "raw_results": results}

        return {"summary": "No relevant product information found.", "raw_results": []}

    def run(self, query_type, query):
        """
        Run the data retrieval agent to fetch information.

        Args:
            query_type (str): Type of data to retrieve ('client' or 'product').
            query (str): The query to search for.

        Returns:
            dict: The retrieved information.
        """
        if not self.client_data or not self.product_data:
            self.load_data()

        if query_type.lower() == "client":
            return self.retrieve_client_info(query)
        elif query_type.lower() == "product":
            return self.retrieve_product_info(query)
        else:
            return {
                "summary": f"Invalid query type: {query_type}. Use 'client' or 'product'.",
                "raw_results": [],
            }
