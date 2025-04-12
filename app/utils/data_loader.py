import os
import pandas as pd
import docx2txt
import json
from config.config import CLIENT_STATE_PATH, PRODUCT_PORTFOLIO_PATH, TRANSCRIPT_PATH
from openai import AzureOpenAI
from dotenv import load_dotenv


class DataLoader:
    """Utility class to load and process data from different file formats."""

    @staticmethod
    def load_client_state():
        """Load client state from Excel file."""
        try:
            if not os.path.exists(CLIENT_STATE_PATH):
                print(f"Warning: Client state file not found at {CLIENT_STATE_PATH}")
                return {}

            # Load the Excel file
            df = pd.read_excel(CLIENT_STATE_PATH)

            # Convert DataFrame to a dictionary format
            client_data = {}

            # Process each row and structure the data
            for _, row in df.iterrows():
                # This is a simplified approach - adjust based on actual Excel structure
                if "Category" in df.columns and "Value" in df.columns:
                    client_data[row["Category"]] = row["Value"]
                else:
                    # If structure is different, convert row to dict
                    row_dict = row.to_dict()
                    client_data.update(row_dict)

            return client_data
        except Exception as e:
            print(f"Error loading client state: {e}")
            return {}

    @staticmethod
    def load_product_portfolio():
        """Extract text from product portfolio document."""
        try:
            if not os.path.exists(PRODUCT_PORTFOLIO_PATH):
                print(
                    f"Warning: Product portfolio file not found at {PRODUCT_PORTFOLIO_PATH}"
                )
                return ""

            # Extract text from DOCX file
            text = docx2txt.process(PRODUCT_PORTFOLIO_PATH)
            return text
        except Exception as e:
            print(f"Error loading product portfolio: {e}")
            return ""

    @staticmethod
    def load_transcript():
        """
        Load transcript text from audio file or cached transcript.
        Uses Azure OpenAI Whisper for transcription if needed.
        """
        # Define the path for the cached transcript
        transcript_cache_path = os.path.join(
            os.path.dirname(TRANSCRIPT_PATH), "transcript.txt"
        )

        # Check if cached transcript exists
        if os.path.exists(transcript_cache_path):
            print(f"Loading cached transcript from {transcript_cache_path}")
            with open(transcript_cache_path, "r") as f:
                return f.read()

        # If no cached transcript, check if audio file exists
        if not os.path.exists(TRANSCRIPT_PATH):
            raise FileNotFoundError(f"Transcript file not found at {TRANSCRIPT_PATH}")

        # Load environment variables
        load_dotenv()

        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_version="2024-02-15-preview",
            azure_endpoint="https://swisshacks-aoai-westeurope.openai.azure.com/",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # Transcribe the audio file
        print(f"Transcribing audio file {TRANSCRIPT_PATH}...")
        with open(TRANSCRIPT_PATH, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper", file=audio_file
            )

        # Extract the text content
        transcript_text = transcript.text

        # Save the transcript to cache
        print(f"Saving transcript to {transcript_cache_path}")
        with open(transcript_cache_path, "w") as f:
            f.write(transcript_text)

        return transcript_text
