from app.agents.data_retrieval_agent import DataRetrievalAgent
from app.agents.dialogue_analysis_agent import DialogueAnalysisAgent
from app.agents.summarization_agent import SummarizationAgent
from app.agents.recommendation_agent import RecommendationAgent
import json
import os
from datetime import datetime


class Orchestrator:
    """Coordinates all agents to process a client-advisor conversation."""

    def __init__(self):
        """Initialize the orchestrator with all required agents."""
        # Initialize all agents
        self.data_retrieval_agent = DataRetrievalAgent()
        self.dialogue_analysis_agent = DialogueAnalysisAgent()
        self.summarization_agent = SummarizationAgent(
            dialogue_analysis_agent=self.dialogue_analysis_agent
        )
        self.recommendation_agent = RecommendationAgent(
            data_retrieval_agent=self.data_retrieval_agent,
            dialogue_analysis_agent=self.dialogue_analysis_agent,
        )

        # Create output directory
        self.output_dir = os.path.abspath("output")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_pipeline(self):
        """
        Run the full pipeline to process the conversation.

        Returns:
            dict: Results from all agents.
        """
        print("\n" + "=" * 50)
        print("Starting conversation analysis pipeline...")
        print("=" * 50)

        # Step 1: Load data
        print("\nStep 1: Loading data...")
        self.data_retrieval_agent.load_data()
        self.dialogue_analysis_agent.load_transcript()

        # Step 2: Run dialogue analysis
        print("\nStep 2: Analyzing dialogue...")
        dialogue_analysis = self.dialogue_analysis_agent.run()
        print(
            f"Found {len(dialogue_analysis.get('topics', []))} topics and performed emotional analysis"
        )

        # Step 3: Create structured summary
        print("\nStep 3: Creating structured summary...")
        summary = self.summarization_agent.run(summary_type="structured")
        print("Summary created with sections: " + ", ".join(summary.keys()))

        # Step 4: Generate recommendations
        print("\nStep 4: Generating recommendations...")
        recommendations = self.recommendation_agent.run(summary=summary)
        print(
            f"Generated {len(recommendations.get('product_recommendations', []))} product recommendations"
        )
        print(f"Identified {len(recommendations.get('unmet_needs', []))} unmet needs")
        print(f"Suggested {len(recommendations.get('next_steps', []))} next steps")

        # Combine all results
        results = {
            "dialogue_analysis": dialogue_analysis,
            "summary": summary,
            "recommendations": recommendations,
        }

        # Save results to file
        self._save_results(results)

        print("\n" + "=" * 50)
        print("Conversation analysis pipeline completed!")
        print("=" * 50)

        return results

    def _save_results(self, results):
        """
        Save results to JSON file.

        Args:
            results (dict): Results from all agents.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_analysis_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filepath}")
        except Exception as e:
            print(f"Error saving results: {e}")

    def generate_report(self, results=None):
        """
        Generate a human-readable report from the results.

        Args:
            results (dict): Optional results from run_pipeline.
                If not provided, run_pipeline will be called.

        Returns:
            str: Formatted report.
        """
        if results is None:
            results = self.run_pipeline()

        summary = results.get("summary", {})
        recommendations = results.get("recommendations", {})

        # Format unmet needs with proper newlines
        unmet_needs = recommendations.get("unmet_needs", [])
        if unmet_needs:
            unmet_needs_formatted = "- " + "\n- ".join(unmet_needs)
        else:
            unmet_needs_formatted = "None identified"

        # Format product recommendations with proper newlines
        product_recommendations = recommendations.get("product_recommendations", [])
        if product_recommendations:
            product_recommendations_formatted = "- " + "\n- ".join(
                product_recommendations
            )
        else:
            product_recommendations_formatted = "None"

        # Format next steps with proper newlines
        next_steps = recommendations.get("next_steps", [])
        if next_steps:
            next_steps_formatted = "- " + "\n- ".join(next_steps)
        else:
            next_steps_formatted = "None"

        # Format the report
        report = f"""
        =======================================
        MEETING ANALYSIS REPORT
        =======================================
        
        CLIENT GOALS & QUESTIONS:
        {summary.get("client_goals", "N/A")}
        
        ADVISOR'S ANALYSIS & RECOMMENDATIONS:
        {summary.get("advisor_recommendations", "N/A")}
        
        ACTION ITEMS & NEXT STEPS:
        {summary.get("action_items", "N/A")}
        
        CLIENT'S REACTIONS & CONCERNS:
        {summary.get("client_reactions", "N/A")}
        
        =======================================
        ADDITIONAL INSIGHTS
        =======================================
        
        UNMET FINANCIAL NEEDS:
        {unmet_needs_formatted}
        
        PRODUCT RECOMMENDATIONS:
        {product_recommendations_formatted}
        
        SUGGESTED NEXT STEPS:
        {next_steps_formatted}
        """

        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"meeting_report_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)

        try:
            with open(filepath, "w") as f:
                f.write(report)
            print(f"\nReport saved to {filepath}")
        except Exception as e:
            print(f"Error saving report: {e}")

        return report
