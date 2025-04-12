import os
import pandas as pd
import docx2txt
import json
from config.config import CLIENT_STATE_PATH, PRODUCT_PORTFOLIO_PATH, TRANSCRIPT_PATH


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
        Load transcript text.
        In a real system, this would use Whisper or another STT system to transcribe from m4a.
        For this prototype, we'll simulate a transcript.
        """
        # In a real implementation, we would use OpenAI's Whisper or another STT service
        # For this prototype, we'll return a simulated transcript
        if not os.path.exists(TRANSCRIPT_PATH):
            print(f"Warning: Transcript file not found at {TRANSCRIPT_PATH}")
            # Return a simulated transcript for testing purposes
            return """
            Advisor: Good morning, Ms. Johnson. Thank you for meeting with me today. How have you been?
            
            Client: Good morning. I've been well, thank you. Just busy with work and family as usual.
            
            Advisor: I understand. Before we dive in, I'd like to review what we discussed in our last meeting. We talked about your retirement goals and the possibility of starting a college fund for your daughter. Has anything changed since then?
            
            Client: Not really, but I've been thinking more about retirement lately. I'm concerned that I'm not saving enough. And yes, I definitely want to set up that college fund soon.
            
            Advisor: Those are important concerns. Let's look at your current retirement savings and see if we need to adjust your contributions. According to my records, you're currently contributing 5% of your salary to your 401(k), is that correct?
            
            Client: Yes, that's right. But I'm wondering if that's enough. Some of my colleagues are contributing much more.
            
            Advisor: Everyone's situation is different, but let's analyze yours. Based on your current salary of $85,000 and your goal to retire at 65, you might want to consider increasing your contribution to at least 10% if possible. This would put you in a better position to meet your retirement goal of $1.5 million.
            
            Client: That makes sense. I think I could increase to 8% now and then maybe go to 10% next year when I expect a raise.
            
            Advisor: That's a sound approach. Incremental increases are often more manageable. Now, regarding the college fund for your daughter, who is 8 years old now, correct?
            
            Client: Yes, she just turned 8 last month.
            
            Advisor: Great. For college savings, I'd recommend considering a 529 plan. It offers tax advantages for education expenses. We could start with a monthly contribution of $300, which would give her a good start for college in 10 years.
            
            Client: That sounds reasonable. I'm also a bit worried about the market right now. Is this a good time to be investing more?
            
            Advisor: That's a common concern. Market timing is difficult even for professionals. What I generally recommend is dollar-cost averaging – investing a fixed amount regularly regardless of market conditions. This approach helps reduce the impact of volatility over time.
            
            Client: I see. That makes me feel better about investing regularly.
            
            Advisor: I'm glad. Now, there's one more area I'd like to discuss. I noticed you don't currently have life insurance beyond the basic policy provided by your employer. Given that you're a single parent, you might want to consider additional coverage to ensure your daughter would be financially secure in any circumstance.
            
            Client: You're right. I've been meaning to look into that. What would you recommend?
            
            Advisor: Based on your situation, I'd suggest a term life insurance policy with coverage of around $500,000 to $750,000. This would help cover your daughter's living expenses and education if needed. The premium would likely be around $30-40 per month given your age and health.
            
            Client: That's more affordable than I expected. Let's definitely explore that option.
            
            Advisor: Excellent. To summarize, we're going to: 1) Increase your 401(k) contribution to 8% now with a plan to go to 10% next year; 2) Set up a 529 college savings plan with a $300 monthly contribution; and 3) Apply for a term life insurance policy. Do you have any questions about these steps?
            
            Client: No, that all sounds good to me. When can we get started on implementing these changes?
            
            Advisor: We can begin right away. I'll prepare the paperwork for the 529 plan and the life insurance application today. You can adjust your 401(k) contribution through your company's HR portal. I'll send you an email with all the details and next steps.
            
            Client: Perfect. Thank you for your help.
            
            Advisor: You're welcome. I'm here to help you reach your financial goals. Let's schedule a follow-up meeting in about three months to review how these changes are working for you. Would that work for you?
            
            Client: Yes, that would be great.
            
            Advisor: Excellent. I'll send you a calendar invitation. Thank you for your time today, and please reach out if you have any questions before our next meeting.
            
            Client: I will. Have a good day.
            
            Advisor: You too. Goodbye.
            """
        else:
            print("In a real implementation, we would transcribe the audio file.")
            # Placeholder for actual transcription logic
            return "This is a placeholder for the actual transcript that would be generated from the audio file."
