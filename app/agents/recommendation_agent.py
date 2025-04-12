from app.agents.base_agent import BaseAgent
from app.utils.data_loader import DataLoader
from app.agents.data_retrieval_agent import DataRetrievalAgent
from app.agents.dialogue_analysis_agent import DialogueAnalysisAgent


class RecommendationAgent(BaseAgent):
    """Agent responsible for suggesting products and next steps based on the conversation."""

    def __init__(
        self, data_retrieval_agent=None, dialogue_analysis_agent=None, *args, **kwargs
    ):
        """Initialize the recommendation agent."""
        super().__init__(*args, **kwargs)
        self.transcript = None
        self.data_retrieval_agent = data_retrieval_agent or DataRetrievalAgent()
        self.dialogue_analysis_agent = (
            dialogue_analysis_agent or DialogueAnalysisAgent()
        )

    def load_transcript(self):
        """Load the transcript from the data loader."""
        print("Loading transcript...")
        self.transcript = DataLoader.load_transcript()
        return self.transcript

    def identify_unmet_needs(self, client_data, topics, emotional_insights):
        """
        Identify unmet financial needs based on the conversation.

        Args:
            client_data (dict): Client information from data retrieval.
            topics (list): Main topics from dialogue analysis.
            emotional_insights (str): Emotional analysis from dialogue analysis.

        Returns:
            list: Identified unmet needs.
        """
        # Convert topics list to string for prompt
        topics_str = "\n".join(topics) if isinstance(topics, list) else topics

        # Convert client data to string
        client_data_str = (
            "\n".join([f"{k}: {v}" for k, v in client_data.items()])
            if isinstance(client_data, dict)
            else client_data
        )

        prompt_template = """
        Analyze the following information about a client's financial situation, the topics discussed
        in their conversation with a financial advisor, and the emotional insights from that conversation.
        
        Client Information:
        {client_data}
        
        Topics Discussed:
        {topics}
        
        Emotional Insights:
        {emotional_insights}
        
        Based on this information, identify any unmet financial needs or gaps in the client's financial portfolio.
        Consider areas that were not adequately addressed in the conversation or products/services that
        the client might benefit from but were not discussed.
        
        List the unmet needs in order of priority, with a brief explanation for each.
        """

        # Format the prompt with the client data, topics, and emotional insights
        formatted_prompt = prompt_template.format(
            client_data=client_data_str,
            topics=topics_str,
            emotional_insights=emotional_insights,
        )

        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a financial advisor specializing in identifying unmet client needs.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        # Get unmet needs from Azure OpenAI
        unmet_needs = self.get_completion(messages)

        # Process into list (in a real system, we'd use more structured output parsing)
        needs_list = [need.strip() for need in unmet_needs.split("\n") if need.strip()]

        return needs_list

    def recommend_products(self, client_data, unmet_needs):
        """
        Recommend products based on unmet needs.

        Args:
            client_data (dict): Client information from data retrieval.
            unmet_needs (list): Identified unmet needs.

        Returns:
            list: Recommended products with justifications.
        """
        # Retrieve product information based on unmet needs
        combined_needs = (
            " ".join(unmet_needs) if isinstance(unmet_needs, list) else unmet_needs
        )
        product_info = self.data_retrieval_agent.run("product", combined_needs)

        # Convert client data to string
        client_data_str = (
            "\n".join([f"{k}: {v}" for k, v in client_data.items()])
            if isinstance(client_data, dict)
            else client_data
        )

        # Convert unmet needs to string
        unmet_needs_str = (
            "\n".join(unmet_needs) if isinstance(unmet_needs, list) else unmet_needs
        )

        prompt_template = """
        Based on the client's information and identified unmet financial needs, recommend specific financial products
        or services that would address these needs. Consider the client's current situation and priorities.
        
        Client Information:
        {client_data}
        
        Unmet Financial Needs:
        {unmet_needs}
        
        Relevant Product Information:
        {product_info}
        
        For each recommendation, provide:
        1. The specific product or service name
        2. How it addresses the client's needs
        3. Why it's a good fit for this particular client
        4. Any considerations or caveats the advisor should mention
        
        List your recommendations in priority order (most important first).
        """

        # Format the prompt with the client data, unmet needs, and product info
        formatted_prompt = prompt_template.format(
            client_data=client_data_str,
            unmet_needs=unmet_needs_str,
            product_info=(
                product_info["summary"]
                if isinstance(product_info, dict)
                else product_info
            ),
        )

        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a financial advisor specializing in product recommendations.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        # Get product recommendations from Azure OpenAI
        recommendations = self.get_completion(messages)

        # Process into list (in a real system, we'd use more structured output parsing)
        recommendation_list = [
            rec.strip() for rec in recommendations.split("\n") if rec.strip()
        ]

        return recommendation_list

    def suggest_next_steps(self, summary, recommendations):
        """
        Suggest next steps for the advisor based on the conversation summary and recommendations.

        Args:
            summary (dict): Structured summary of the conversation.
            recommendations (list): Product recommendations.

        Returns:
            list: Suggested next steps with timeframes.
        """
        # Extract summary sections
        client_goals = summary.get("client_goals", "")
        advisor_recommendations = summary.get("advisor_recommendations", "")
        action_items = summary.get("action_items", "")

        # Convert recommendations to string
        recommendations_str = (
            "\n".join(recommendations)
            if isinstance(recommendations, list)
            else recommendations
        )

        prompt_template = """
        Based on the conversation summary and product recommendations, suggest a prioritized list of next steps
        for the financial advisor to take with this client. Include specific actions, their purpose, and suggested timeframes.
        
        Client's Goals/Questions:
        {client_goals}
        
        Advisor's Recommendations from Meeting:
        {advisor_recommendations}
        
        Action Items Already Identified:
        {action_items}
        
        Additional Product Recommendations:
        {recommendations}
        
        Suggest a comprehensive list of next steps that combines the action items already identified in the meeting
        with additional steps based on the recommended products. For each step, specify:
        
        1. The action to take
        2. Why it's important
        3. When it should be done (timeframe)
        4. Any prerequisites or dependencies
        
        Format each step as: "ACTION: [description] - TIMEFRAME: [when] - PURPOSE: [why]"
        """

        # Format the prompt with the summary sections and recommendations
        formatted_prompt = prompt_template.format(
            client_goals=client_goals,
            advisor_recommendations=advisor_recommendations,
            action_items=action_items,
            recommendations=recommendations_str,
        )

        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a financial advisor specializing in action planning.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        # Get next steps from Azure OpenAI
        next_steps = self.get_completion(messages)

        # Process into list
        next_steps_list = [
            step.strip() for step in next_steps.split("\n") if step.strip()
        ]

        return next_steps_list

    def run(self, summary=None):
        """
        Run the recommendation agent to generate recommendations.

        Args:
            summary (dict): Optional structured summary from the summarization agent.

        Returns:
            dict: The recommendations and next steps.
        """
        if not self.transcript:
            self.load_transcript()

        # Get dialogue analysis
        print("Running dialogue analysis...")
        dialogue_analysis = self.dialogue_analysis_agent.run()

        # Get client data
        print("Retrieving client data...")
        client_data = self.data_retrieval_agent.run(
            "client", "client profile and financial situation"
        )

        # Extract topics and emotional insights from dialogue analysis
        topics = dialogue_analysis.get("topics", [])
        emotional_insights = dialogue_analysis.get("emotions", "")

        # Identify unmet needs
        print("Identifying unmet needs...")
        unmet_needs = self.identify_unmet_needs(
            client_data["summary"] if isinstance(client_data, dict) else client_data,
            topics,
            emotional_insights,
        )

        # Recommend products
        print("Generating product recommendations...")
        recommendations = self.recommend_products(
            client_data["summary"] if isinstance(client_data, dict) else client_data,
            unmet_needs,
        )

        # If summary is provided, suggest next steps
        next_steps = []
        if summary:
            print("Suggesting next steps...")
            next_steps = self.suggest_next_steps(summary, recommendations)

        return {
            "unmet_needs": unmet_needs,
            "product_recommendations": recommendations,
            "next_steps": next_steps,
        }
