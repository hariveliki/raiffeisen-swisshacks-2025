from app.agents.base_agent import BaseAgent
from app.utils.data_loader import DataLoader
from app.agents.dialogue_analysis_agent import DialogueAnalysisAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


class SummarizationAgent(BaseAgent):
    """Agent responsible for creating structured summaries of the client-advisor conversation."""

    def __init__(self, dialogue_analysis_agent=None, *args, **kwargs):
        """Initialize the summarization agent."""
        super().__init__(*args, **kwargs)
        self.transcript = None
        self.dialogue_analysis_agent = (
            dialogue_analysis_agent or DialogueAnalysisAgent()
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def load_transcript(self):
        """Load the transcript from the data loader."""
        print("Loading transcript...")
        self.transcript = DataLoader.load_transcript()
        return self.transcript

    def summarize_chunk(self, chunk):
        """
        Summarize a single chunk of the transcript.

        Args:
            chunk (str): A section of the transcript.

        Returns:
            str: Summary of the chunk.
        """
        prompt_template = """
        Summarize the following excerpt from a conversation between a financial advisor and a client.
        Focus on key financial points, questions, concerns, and decisions made.
        
        Conversation Excerpt:
        {chunk}
        
        Summary:
        """

        # Format the prompt with the chunk
        formatted_prompt = prompt_template.format(chunk=chunk)

        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a financial conversation summarizer.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        # Get summary from Azure OpenAI
        chunk_summary = self.get_completion(messages)

        return chunk_summary

    def summarize_transcript_chunks(self):
        """
        Split the transcript into chunks and summarize each chunk.
        Then combine the chunk summaries.

        Returns:
            str: Combined summary of the transcript.
        """
        if not self.transcript:
            self.load_transcript()

        # Split the transcript into chunks
        chunks = self.text_splitter.split_text(self.transcript)
        print(f"Transcript split into {len(chunks)} chunks.")

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Summarizing chunk {i+1}/{len(chunks)}...")
            summary = self.summarize_chunk(chunk)
            chunk_summaries.append(summary)

        # Combine chunk summaries
        combined_summaries = "\n\n".join(chunk_summaries)

        # Create a final unified summary
        prompt_template = """
        Below are summaries of different parts of a conversation between a financial advisor and a client.
        Create a coherent, concise overall summary of the entire conversation while preserving all key details.
        
        Chunk Summaries:
        {combined_summaries}
        
        Overall Summary:
        """

        # Format the prompt with the combined summaries
        formatted_prompt = prompt_template.format(combined_summaries=combined_summaries)

        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a financial conversation summarizer.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        # Get final summary from Azure OpenAI
        final_summary = self.get_completion(messages)

        return final_summary

    def create_structured_summary(self, dialogue_analysis=None):
        """
        Create a structured summary with specific sections.

        Args:
            dialogue_analysis (dict): Results from the dialogue analysis agent.

        Returns:
            dict: Structured summary with sections.
        """
        if not self.transcript:
            self.load_transcript()

        # Get dialogue analysis if not provided
        if not dialogue_analysis:
            print("Running dialogue analysis...")
            dialogue_analysis = self.dialogue_analysis_agent.run()

        # Get emotional insights and topics from dialogue analysis
        emotional_insights = dialogue_analysis.get("emotions", "")
        topics = dialogue_analysis.get("topics", [])

        # Create structured sections
        prompt_template = """
        Create a structured summary of the following conversation between a financial advisor and a client.
        The summary should be organized into specific sections as outlined below.
        
        Transcript:
        {transcript}
        
        Emotional Analysis:
        {emotional_insights}
        
        Main Topics:
        {topics}
        
        Please provide the summary in the following format:
        
        **Client's Goals/Questions:**
        (List the main goals and questions the client had)
        
        **Advisor's Analysis & Recommendations:**
        (Summarize the key analyses and specific recommendations the advisor provided)
        
        **Action Items / Next Steps:**
        (List concrete next steps agreed upon, with any deadlines or timelines)
        
        **Client's Reactions/Concerns:**
        (Summarize how the client responded to the advice, any concerns raised)
        
        Keep each section concise but comprehensive. Use bullet points where appropriate.
        """

        # Format topics as a string
        topics_str = "\n".join(topics) if isinstance(topics, list) else topics

        # Format the prompt with the transcript, emotional insights, and topics
        formatted_prompt = prompt_template.format(
            transcript=self.transcript,
            emotional_insights=emotional_insights,
            topics=topics_str,
        )

        # Create messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a financial conversation analyst specializing in structured summaries.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        # Get structured summary from Azure OpenAI
        structured_output = self.get_completion(messages)

        # Process the output to extract sections
        sections = {}

        # Parse the sections (simple approach - in a real system we'd use regex or structured output)
        if "**Client's Goals/Questions:**" in structured_output:
            parts = structured_output.split("**Client's Goals/Questions:**")[1].split(
                "**Advisor's Analysis"
            )[0]
            sections["client_goals"] = parts.strip()

        if "**Advisor's Analysis & Recommendations:**" in structured_output:
            parts = structured_output.split(
                "**Advisor's Analysis & Recommendations:**"
            )[1].split("**Action Items")[0]
            sections["advisor_recommendations"] = parts.strip()

        if "**Action Items / Next Steps:**" in structured_output:
            parts = structured_output.split("**Action Items / Next Steps:**")[1].split(
                "**Client's Reactions"
            )[0]
            sections["action_items"] = parts.strip()

        if "**Client's Reactions/Concerns:**" in structured_output:
            parts = structured_output.split("**Client's Reactions/Concerns:**")[1]
            sections["client_reactions"] = parts.strip()

        return sections

    def run(self, summary_type="structured"):
        """
        Run the summarization agent to create a summary of the transcript.

        Args:
            summary_type (str): Type of summary to create
                ('chunks', 'structured', or 'full').

        Returns:
            dict or str: The summary results.
        """
        if not self.transcript:
            self.load_transcript()

        if summary_type == "chunks":
            return self.summarize_transcript_chunks()
        elif summary_type == "structured":
            return self.create_structured_summary()
        elif summary_type == "full":
            # Get both types of summaries
            chunk_summary = self.summarize_transcript_chunks()
            structured_summary = self.create_structured_summary()

            return {
                "chunk_summary": chunk_summary,
                "structured_summary": structured_summary,
            }
        else:
            raise ValueError(
                f"Invalid summary type: {summary_type}. Must be 'chunks', 'structured', or 'full'."
            )
