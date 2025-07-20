🚀 Features

🔄 Cache reuse: Query or retrain on the same PDF

⏳ Progress bars: Live feedback during embeddings

📊 Token reporting: Shows total tokens & chunk count after chunking

📝 Transcript management: Auto-numbered logs to avoid overwriting

🔑 Prerequisites

Python 3.8+

OpenAI API key: Place your key in a .env file in the repo root like mine:

# .env
OPENAI_API_KEY=sk-...

Install dependencies:

pip install -r requirements.txt

⚙️ Installation

Clone this repo:

git clone https://github.com/MaxBlanksby/BookSummerizationApp

cd BookSummerizationApp

Create a .env with your OpenAI key (as above).

Install:

pip install -r requirements.txt

▶️ Usage

python3 pdf_chatbot.py path/to/your.pdf

Select to query cache or retrain

Wait for chunking & embedding (progress shown)

Chat! Type questions; exit to quit

Find transcripts in Chatbot_Interactions/<pdf_name>/

