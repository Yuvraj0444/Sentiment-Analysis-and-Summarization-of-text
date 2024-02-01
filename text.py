import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Download the VADER lexicon and stopwords
nltk.download('vader_lexicon')
nltk.download('punkt')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Prompt the user to enter the text for analysis
text = input("Enter the text for analysis: ")

# Text summarization
parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer(Stemmer("english"))
summarizer.stop_words = get_stop_words("english")
summary = list(summarizer(parser.document, 2))  # Get a 2-sentence summary

# Get the full summary as a string
full_summary = " ".join(str(sentence) for sentence in summary)

# Get the polarity scores
scores = sid.polarity_scores(text)

# Interpret the sentiment polarity
if scores['compound'] > 0:
    sentiment = "Positive"
elif scores['compound'] < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Print the summary and sentiment
print("Full Summary:")
print(full_summary)
print("\nSentiment:", sentiment)