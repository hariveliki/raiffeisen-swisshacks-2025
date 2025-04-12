from app.agents.base_agent import BaseAgent
from app.utils.data_loader import DataLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


class DialogueAnalysisAgent(BaseAgent):
    """Agent responsible for analyzing the transcript of client-advisor conversation."""

    def __init__(self, *args, **kwargs):
        """Initialize the dialogue analysis agent."""
        super().__init__(*args, **kwargs)
        self.transcript = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

    def load_transcript(self):
        """Load the transcript from the data loader."""
        print("Loading transcript...")
        self.transcript = DataLoader.load_transcript()
        return self.transcript

    def split_transcript(self):
        """Split the transcript into manageable chunks."""
        if not self.transcript:
            self.load_transcript()

        print("Splitting transcript into chunks...")
        chunks = self.text_splitter.split_text(self.transcript)
        print(f"Transcript split into {len(chunks)} chunks.")
        return chunks

    def analyze_emotions(self):
        """
        Analyze the emotional cues and client sentiment in the transcript.

        Returns:
            str: Summary of emotional analysis.
        """
        if not self.transcript:
            self.load_transcript()

        # Prompt for emotion analysis using Chain-of-Thought
        prompt_template = """
        You are analyzing a transcript of a conversation between a financial advisor and a client. 
        Use Chain-of-Thought reasoning to identify the emotional cues and sentiment in the conversation.
        
        First, identify all statements that indicate the client's emotions or attitudes.
        Then, deduce the client's underlying concerns or needs from those.
        Finally, summarize these insights in a concise way.
        
        Transcript:
        {transcript}
        
        Step 1 - Identify statements indicating emotions:
        
        Step 2 - Deduce underlying concerns/needs:
        
        Step 3 - Summarize emotional insights:
        """

        prompt = PromptTemplate(
            input_variables=["transcript"], template=prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Get analysis from LLM
        analysis = chain.run(transcript=self.transcript)

        # Extract just the summary from the Chain-of-Thought output
        # (In a real system, we might use more structured output parsing)
        if "Step 3 - Summarize emotional insights:" in analysis:
            summary = analysis.split("Step 3 - Summarize emotional insights:")[
                1
            ].strip()
        else:
            summary = analysis

        return summary

    def extract_topics(self):
        """
        Extract the main topics discussed in the conversation.

        Returns:
            list: List of main topics.
        """
        if not self.transcript:
            self.load_transcript()

        prompt_template = """
        Analyze the following transcript of a conversation between a financial advisor and a client.
        Identify the main financial topics discussed and list them in order of importance.
        For each topic, provide a brief one-line description.
        
        Transcript:
        {transcript}
        
        Main Topics (in order of importance):
        """

        prompt = PromptTemplate(
            input_variables=["transcript"], template=prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Get topics from LLM
        topics = chain.run(transcript=self.transcript)

        # Process the output to return a structured list
        # (In a real system, we'd use more structured output parsing)
        topic_list = [topic.strip() for topic in topics.split("\n") if topic.strip()]

        return topic_list

    def extract_questions(self):
        """
        Extract questions asked by the client.

        Returns:
            list: List of client questions.
        """
        if not self.transcript:
            self.load_transcript()

        prompt_template = """
        Review the following transcript of a conversation between a financial advisor and a client.
        Extract all questions asked by the client, both explicit questions and implied questions.
        For implied questions, explain what indicates the client's curiosity or concern.
        
        Transcript:
        {transcript}
        
        Client Questions:
        """

        prompt = PromptTemplate(
            input_variables=["transcript"], template=prompt_template
        )

        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Get questions from LLM
        questions = chain.run(transcript=self.transcript)

        # Process the output to return a structured list
        question_list = [q.strip() for q in questions.split("\n") if q.strip()]

        return question_list

    def run(self, analysis_type="full"):
        """
        Run the dialogue analysis agent to analyze the transcript.

        Args:
            analysis_type (str): Type of analysis to perform
                ('emotions', 'topics', 'questions', or 'full').

        Returns:
            dict: The analysis results.
        """
        if not self.transcript:
            self.load_transcript()

        results = {}

        if analysis_type == "emotions" or analysis_type == "full":
            results["emotions"] = self.analyze_emotions()

        if analysis_type == "topics" or analysis_type == "full":
            results["topics"] = self.extract_topics()

        if analysis_type == "questions" or analysis_type == "full":
            results["questions"] = self.extract_questions()

        return results
