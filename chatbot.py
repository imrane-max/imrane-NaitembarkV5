#!/usr/bin/env python3
"""
Portfolio AI Chatbot with Real AI & Machine Learning
- Real AI: Ollama (free local LLM) or OpenAI API for generative responses
- ML Fallback: TF-IDF + N-grams + BM25 + Cosine Similarity (pure Python, no deps)
Features: LLM, TF-IDF, N-grams, BM25 ranking, ensemble matching, online learning
"""

import argparse
import os
import random
import re
import math
import sys
from datetime import datetime
from collections import defaultdict, Counter

# Optional: Real AI backends
OLLAMA_AVAILABLE = False
OPENAI_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    pass
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass


# System prompt for Real AI - gives context about the portfolio
PORTFOLIO_SYSTEM_PROMPT = """You are the AI assistant for Imrane Naitembark's portfolio. You help visitors learn about his work.

ABOUT IMRANE:
- Web developer, game developer, AI/ML enthusiast
- Skills: HTML, CSS, JavaScript, Python, Godot GDScript, Neural Networks, TF-IDF, Canvas API
- Projects: 12+ games (Snake, 2048, Memory Match, Reaction Time, Tic Tac Toe, ML Digit Classifier, ML RPS Learner, Godot games Fragime & Hamood)
- Contact: imrane2015su@gmail.com, github.com/imrane-max

Keep responses concise, helpful, and friendly. Use emojis occasionally. Answer questions about his portfolio, skills, projects, games, and contact info."""


class TFIDF:
    """TF-IDF Vectorizer with N-grams for text similarity (Machine Learning)"""
    
    def __init__(self, use_ngrams=True, ngram_range=(1, 2)):
        self.documents = []
        self.vocabulary = set()
        self.idf = {}
        self.doc_vectors = []
        self.use_ngrams = use_ngrams
        self.ngram_range = ngram_range
    
    def tokenize(self, text):
        """Simple tokenization - lowercase and split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'can', 'to',
                     'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
                     'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'between', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'when', 'where', 'why', 'how',
                     'all', 'each', 'few', 'more', 'most', 'other', 'some',
                     'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'this', 'that', 'these',
                     'those', 'am', 'it', 'its', 'i', 'me', 'my', 'myself',
                     'we', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him',
                     'his', 'she', 'her', 'hers', 'they', 'them', 'their',
                     'what', 'which', 'who', 'whom'}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def _get_ngrams(self, tokens):
        """Generate n-grams (unigrams + bigrams) for richer ML features"""
        ngrams = list(tokens)
        if self.use_ngrams and self.ngram_range[1] >= 2:
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                ngrams.append(bigram)
        return ngrams
    
    def fit(self, documents):
        """Build vocabulary and compute IDF values"""
        self.documents = documents
        doc_count = len(documents)
        term_doc_freq = defaultdict(int)
        self.doc_term_freqs = []  # Raw term freqs per doc for BM25
        
        # Build vocabulary and document frequency (with n-grams)
        for doc in documents:
            tokens = self.tokenize(doc)
            ngrams = self._get_ngrams(tokens)
            self.vocabulary.update(ngrams)
            self.doc_term_freqs.append(Counter(ngrams))
            for term in set(ngrams):
                term_doc_freq[term] += 1
        
        # Compute IDF: log(N / df) + 1 (smoothed)
        for term in self.vocabulary:
            self.idf[term] = math.log(doc_count / (term_doc_freq[term] + 1)) + 1
        
        # Compute document vectors
        self.doc_vectors = [self._compute_tfidf(doc) for doc in documents]
    
    def _compute_tfidf(self, text):
        """Compute TF-IDF vector for a text (with n-grams)"""
        tokens = self.tokenize(text)
        if not tokens:
            return {}
        
        # N-gram features for richer ML representation
        ngrams = self._get_ngrams(tokens)
        tf = Counter(ngrams)
        max_tf = max(tf.values()) if tf else 1
        
        # TF-IDF vector
        vector = {}
        for term, count in tf.items():
            if term in self.idf:
                vector[term] = (count / max_tf) * self.idf[term]
        return vector
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0.0
        
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        if not common_terms:
            return 0.0
        
        # Dot product
        dot_product = sum(vec1[t] * vec2[t] for t in common_terms)
        
        # Magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def bm25_score(self, query, doc_idx, k1=1.5, b=0.75):
        """BM25 ranking function - ML algorithm used by search engines"""
        query_tokens = self._get_ngrams(self.tokenize(query))
        if not query_tokens:
            return 0.0

        doc_tf = self.doc_term_freqs[doc_idx]
        n_docs = len(self.documents)
        doc_lengths = [sum(dtf.values()) for dtf in self.doc_term_freqs]
        avg_dl = sum(doc_lengths) / max(n_docs, 1)
        doc_len = doc_lengths[doc_idx] or 1

        score = 0.0
        for term in set(query_tokens):
            if term not in doc_tf or term not in self.idf:
                continue
            tf = doc_tf[term]
            df = sum(1 for dtf in self.doc_term_freqs if term in dtf)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_dl))

        return max(0.0, min(1.0, score / 15))  # Normalize to 0-1 range

    def find_most_similar(self, query, threshold=0.1):
        """Find most similar document: ensemble of Cosine Similarity + BM25 (ML)"""
        query_vector = self._compute_tfidf(query)

        best_idx = -1
        best_score = threshold

        for idx, doc_vector in enumerate(self.doc_vectors):
            cos_sim = self.cosine_similarity(query_vector, doc_vector)
            bm25 = self.bm25_score(query, idx)
            # Ensemble: combine cosine + BM25 for better ML matching
            score = 0.7 * cos_sim + 0.3 * min(1.0, bm25)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score


class RealAI:
    """Real AI backend: Ollama (free local) or OpenAI API"""

    def __init__(self, backend="ollama", model=None):
        self.backend = backend.lower()
        self.model = model or ("llama3.2" if self.backend == "ollama" else "gpt-4o-mini")
        self.client = None
        self.messages = [{"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT}]
        self.available = False

        if self.backend == "ollama" and OLLAMA_AVAILABLE:
            try:
                ollama.list()  # Test connection
                self.available = True
                self.model = model or self._get_available_model()
            except Exception:
                pass
        elif self.backend == "openai" and OPENAI_AVAILABLE:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.available = True
                self.model = model or "gpt-4o-mini"

    def _get_available_model(self):
        """Get first available Ollama model or default"""
        try:
            models = ollama.list()
            if models.get("models"):
                return models["models"][0]["name"].split(":")[0]
        except Exception:
            pass
        return "llama3.2"

    def chat(self, user_input):
        """Get response from Real AI. Returns (response, confidence) or (None, 0) on failure."""
        if not self.available:
            return None, 0.0

        self.messages.append({"role": "user", "content": user_input})

        try:
            if self.backend == "ollama":
                response = ollama.chat(model=self.model, messages=self.messages)
                content = response["message"]["content"]
            else:  # openai
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages
                )
                content = response.choices[0].message.content

            self.messages.append({"role": "assistant", "content": content})
            return content.strip(), 1.0  # Real AI = high confidence

        except Exception as e:
            # Remove failed user message so we can retry
            self.messages.pop()
            return None, 0.0


class PortfolioChatbot:
    def __init__(self, use_real_ai=False, real_ai_backend="ollama", real_ai_model=None):
        self.name = "Portfolio Real AI" if use_real_ai else "Portfolio ML AI"
        self.version = "3.0"
        self.conversation_history = []
        self.response_count = 0
        self.learned_responses = {}  # For learning from conversations
        self.use_real_ai = use_real_ai
        self.real_ai = RealAI(backend=real_ai_backend, model=real_ai_model) if use_real_ai else None
        
        # ML-based knowledge base: training phrases -> responses
        self.ml_knowledge = [
            # Greetings
            ("hello hi hey greetings hola welcome", [
                'Hello! 👋 Welcome to my portfolio. What would you like to know?',
                'Hi there! 😊 I\'m here to help. Ask about my projects, skills, or games!',
                'Greetings! Welcome. Feel free to ask anything about my work!'
            ]),
            
            # Projects
            ("project projects game games build built code portfolio application work", [
                '🎮 I have built 12+ games including:\n- Snake, 2048, Rock Paper Scissors AI\n- Memory Match, Reaction Time, Tic Tac Toe\n- ML Digit Classifier, ML RPS Learner\n- Plus Godot games: Fragime (FPS) and Hamood (Platform)',
                '📦 My portfolio features interactive games with vanilla JavaScript, Canvas API, and Godot Engine. All playable in the browser!',
                '🎯 Check the Games section for 12+ playable games, from simple classics to AI-powered experiences!'
            ]),
            
            # Skills/Tech
            ("skill skills technology tech language languages python javascript java html css programming developer", [
                '💻 My skill set includes:\n- Frontend: HTML, CSS, JavaScript (ES6+), Canvas API\n- Backend: Python, Node.js\n- Game Dev: Godot Engine, GDScript\n- AI/ML: Neural Networks, TF-IDF, Pattern Recognition\n- Web: WebAssembly, Responsive Design',
                '🛠️ I work with: JavaScript, Python, Godot GDScript, HTML5, CSS3, Canvas, Machine Learning, Web APIs!',
                '⚙️ Tech stack: Vanilla JS, Python (ML/Data), Godot, WebAssembly, TensorFlow concepts, Neural Networks'
            ]),
            
            # AI/ML
            ("ai artificial intelligence machine learning ml neural network deep learning algorithm tfidf cosine", [
                '🤖 AI/ML is my passion! I\'ve built:\n- Neural Network Trainer (visualization)\n- TF-IDF Chatbot (NLP)\n- ML Digit Classifier (pattern recognition)\n- ML RPS Learner (adaptive AI)\nCheck the Games section to try them!',
                '🧠 I use Machine Learning techniques like:\n- TF-IDF for text analysis\n- Cosine Similarity for matching\n- Neural Networks with backpropagation\n- Pattern Recognition and Classification',
                '🎓 This chatbot uses TF-IDF vectorization and cosine similarity - real ML algorithms for intelligent matching!'
            ]),
            
            # Games/Play
            ("play playing gaming entertainment fun interactive demo try test", [
                '🎮 You can play these games right now:\n✓ Classic Games: Snake, 2048, Memory Match\n✓ Reaction Time Test\n✓ AI Games: RPS against learning AI, Tic Tac Toe with AI\n✓ ML Games: Digit Classifier, ML Pattern Learner',
                '🕹️ All games are built with vanilla JavaScript - no frameworks!\nJust pure code, Canvas API, and Web APIs.',
                '🎯 Games range from simple classics to complex AI opponents. Try them all!'
            ]),
            
            # Contact
            ("contact email reach message phone discord social connect hire", [
                '📧 Contact Information:\nEmail: imrane2015su@gmail.com\nGithub: github.com/imrane-max\nFeel free to reach out for collaborations!',
                '💌 Email me at: imrane2015su@gmail.com\nI\'d love to discuss projects, opportunities, or just chat about code!',
                '🤝 Best way to reach me: imrane2015su@gmail.com\nAlways happy to connect!'
            ]),
            
            # Timeline
            ("time duration long much weeks days timeline schedule estimate", [
                '⏱️ Project Timelines:\n- Simple games: 1 week\n- Complex games: 2-3 weeks\n- AI features: 2-3 weeks\n- Full portfolio: 3-4 weeks',
                '📅 Most projects take 2-3 weeks. Games are faster (1-2 weeks), larger projects 3-4 weeks.',
                '⌛ Development varies: simple 1 week, medium 2-3 weeks, complex 3-4 weeks'
            ]),
            
            # Help
            ("help support assist what can you do how use guide explain", [
                '🆘 I can help with:\n✓ Projects and games info\n✓ Skills and technologies\n✓ Contact information\n✓ AI/ML explanations\nJust ask anything!',
                '💡 Ask me about: projects, skills, games, AI/ML, or contact info!',
                '🎯 I use TF-IDF machine learning to understand your questions intelligently!'
            ]),
            
            # Thanks
            ("thanks thank appreciate grateful awesome great nice cool", [
                '😊 You\'re welcome! Happy to help. Any other questions?',
                '🙏 Thanks! Feel free to explore the portfolio.',
                '💖 Glad I could help! Let me know if you need anything else!'
            ]),
            
            # Goodbye
            ("bye goodbye see you farewell exit quit leave later", [
                '👋 Goodbye! Thanks for visiting. Try the games!',
                '🚀 See you later! Don\'t forget the games section!',
                '✨ Bye! Hope you enjoyed learning about my work!'
            ]),
            
            # About this chatbot
            ("chatbot bot how work algorithm explain yourself about you", [
                '🤖 I\'m powered by TF-IDF (Term Frequency-Inverse Document Frequency) machine learning!\nI convert text to vectors and use cosine similarity to find the best matching response.',
                '🧠 My algorithm:\n1. Tokenize your input\n2. Compute TF-IDF vectors\n3. Calculate cosine similarity\n4. Return the best match!',
                '📊 I use real NLP: stopword removal, TF-IDF weighting, and vector similarity. Not just keyword matching!'
            ])
        ]
        
        # Initialize TF-IDF with training data
        self.tfidf = TFIDF()
        self._train_tfidf()
    
    def _train_tfidf(self):
        """Train the TF-IDF model on knowledge base"""
        training_docs = [phrase for phrase, _ in self.ml_knowledge]
        # Add learned responses to training
        for phrase in self.learned_responses.keys():
            training_docs.append(phrase)
        self.tfidf.fit(training_docs)
    
    def learn(self, user_input, response):
        """Learn a new response for a given input"""
        processed = user_input.lower().strip()
        if processed not in self.learned_responses:
            self.learned_responses[processed] = []
        self.learned_responses[processed].append(response)
        # Retrain with new knowledge
        self._train_tfidf()
        return f"✅ Learned! I'll remember that '{user_input[:30]}...' relates to your response."
    
    def preprocess_input(self, user_input):
        """Clean and normalize user input"""
        user_input = user_input.strip()
        user_input = user_input.lower()
        return user_input
    
    def find_response(self, user_input):
        """Find best matching response using ML: TF-IDF + N-grams + BM25 ensemble"""
        processed_input = self.preprocess_input(user_input)
        
        # Use TF-IDF to find most similar training phrase
        best_idx, confidence = self.tfidf.find_most_similar(processed_input, threshold=0.15)
        
        if best_idx >= 0:
            # Check if it's a learned response first
            num_knowledge = len(self.ml_knowledge)
            if best_idx >= num_knowledge:
                # It's a learned response
                learned_idx = best_idx - num_knowledge
                learned_phrase = list(self.learned_responses.keys())[learned_idx]
                responses = self.learned_responses[learned_phrase]
            else:
                # It's from the original knowledge base
                _, responses = self.ml_knowledge[best_idx]
            
            response = random.choice(responses)
            return response, confidence, True
        
        # Fallback responses if no good match found
        fallback_responses = [
            '🤔 Interesting question! Try asking about my projects, skills, or games.',
            '💭 I\'m not sure about that. Ask me about AI/ML, games, or contact info!',
            '✨ My ML model couldn\'t find a good match. Try rephrasing or ask about my portfolio!',
            '🎯 I use TF-IDF + BM25 - try keywords like "skills", "games", "AI", or "contact"!'
        ]
        
        return random.choice(fallback_responses), 0.0, False
    
    def chat(self, user_input):
        """Main chat method: Real AI first (if enabled), else TF-IDF ML"""
        response = None
        confidence = 0.0
        matched = False
        used_real_ai = False

        # Try Real AI first when enabled and available
        if self.use_real_ai and self.real_ai and self.real_ai.available:
            response, confidence = self.real_ai.chat(user_input)
            if response:
                matched = True
                used_real_ai = True

        # Fallback to TF-IDF ML
        if response is None:
            response, confidence, matched = self.find_response(user_input)

        # Store in history with confidence score
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat(),
            'matched': matched,
            'confidence': confidence,
            'real_ai': used_real_ai
        })
        self.response_count += 1

        return response, confidence
    
    def get_stats(self):
        """Get conversation statistics with ML metrics"""
        total_exchanges = len(self.conversation_history)
        matched_responses = sum(1 for item in self.conversation_history if item['matched'])
        match_rate = (matched_responses / total_exchanges * 100) if total_exchanges > 0 else 0
        
        # Calculate average confidence
        confidences = [item.get('confidence', 0) for item in self.conversation_history]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_messages': total_exchanges,
            'matched_responses': matched_responses,
            'match_rate': f'{match_rate:.1f}%',
            'avg_confidence': f'{avg_confidence:.2f}',
            'vocabulary_size': len(self.tfidf.vocabulary),
            'learned_responses': len(self.learned_responses),
            'conversations': self.response_count
        }
    
    def show_help(self):
        """Display help information"""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║      Portfolio ML AI Chatbot - Command Help              ║
╚════════════════════════════════════════════════════════════╝

COMMANDS:
  /help       - Show this help message
  /stats      - Show ML statistics and metrics
  /history    - Show conversation history
  /learn      - Teach me a new response (usage: /learn)
  /clear      - Clear conversation history
  /quit       - Exit the chatbot
  /about      - About this ML chatbot

TOPICS YOU CAN ASK ABOUT:
  ✓ Projects & Games
  ✓ Skills & Technologies  
  ✓ AI & Machine Learning
  ✓ Contact Information
  ✓ How this chatbot works

ML FEATURES:
  🧠 TF-IDF Vectorization + N-grams (bigrams)
  📊 BM25 Ranking (search engine algorithm)
  📐 Cosine Similarity + Ensemble matching
  🎓 Online learning from conversations
  📈 Confidence scoring

Just type naturally - I use ML to understand! 😊
════════════════════════════════════════════════════════════
        """
        return help_text
    
    def show_history(self):
        """Display conversation history"""
        if not self.conversation_history:
            return "No conversation history yet!\n"
        
        history_text = "\n📜 CONVERSATION HISTORY:\n"
        history_text += "=" * 50 + "\n"
        
        for i, exchange in enumerate(self.conversation_history, 1):
            history_text += f"\n[{i}] User: {exchange['user']}\n"
            history_text += f"    Bot:  {exchange['bot']}\n"
        
        history_text += "\n" + "=" * 50 + "\n"
        return history_text
    
    def show_about(self):
        """Show about information"""
        about_text = f"""
╔══════════════════════════════════════════════════╗
║       {self.name} v{self.version}                    ║
╚══════════════════════════════════════════════════╝

A Machine Learning powered chatbot for portfolio inquiries.

ML ALGORITHMS:
  🧠 TF-IDF (Term Frequency-Inverse Document Frequency)
  📝 N-grams (unigrams + bigrams) for richer features
  📊 BM25 ranking (used by search engines)
  📐 Cosine Similarity + Ensemble matching
  🎓 Online learning from conversations
  🔤 Stopword removal and tokenization

FEATURES:
  ✓ Intelligent response matching
  ✓ Confidence scoring (0.0 - 1.0)
  ✓ Knowledge base with 11 topics
  ✓ Learning capability (/learn)
  ✓ Conversation history & stats

VOCABULARY: {len(self.tfidf.vocabulary)} terms
LEARNED: {len(self.learned_responses)} custom responses

BUILT WITH: Python 3 (no external libs)
CREATOR: Imrane
PURPOSE: Portfolio ML Assistant

Type /help for commands
══════════════════════════════════════════════════
        """
        return about_text


def main():
    """Main conversation loop - Llama (Ollama) or ML"""
    parser = argparse.ArgumentParser(description='Portfolio AI Chatbot')
    parser.add_argument('--ollama', action='store_true', help='Use free Llama model via Ollama (requires: pip install ollama, ollama pull llama3.2)')
    parser.add_argument('--model', default='llama3.2', help='Ollama model name (default: llama3.2)')
    args = parser.parse_args()

    use_llama = args.ollama and OLLAMA_AVAILABLE
    chatbot = PortfolioChatbot(use_real_ai=use_llama, real_ai_backend='ollama', real_ai_model=args.model)

    if use_llama and chatbot.real_ai.available:
        print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   🦙 Portfolio AI Chatbot - Llama FREE Model!             ║
║                                                            ║
║   Powered by Ollama (Llama) - Real AI responses!          ║
║   Ask me about projects, skills, games, or AI/ML!         ║
║   Type /help for commands or just start chatting!         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """)
        print(f"🦙 Llama model: {chatbot.real_ai.model} (Ollama)")
    else:
        print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   Welcome to the Portfolio ML AI Chatbot! 🤖🧠            ║
║                                                            ║
║   ML: TF-IDF + N-grams + BM25 + Cosine Similarity         ║
║   Use --ollama for free Llama AI (pip install ollama)     ║
║   Type /help for commands or just start chatting!         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
        """)
        print(f"🧠 ML Model loaded with {len(chatbot.tfidf.vocabulary)} vocabulary terms")
    
    learning_mode = False
    learn_phrase = ""
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("💭 Please say something!")
                continue
            
            # Handle learning mode
            if learning_mode:
                result = chatbot.learn(learn_phrase, user_input)
                print(f"\n🤖 Bot: {result}")
                learning_mode = False
                learn_phrase = ""
                continue
            
            # Handle commands
            if user_input.lower() == '/quit':
                print("\n👋 Goodbye! Thanks for chatting!")
                stats = chatbot.get_stats()
                print(f"\n📊 ML Stats: {stats['conversations']} conversations, "
                      f"{stats['match_rate']} match rate, "
                      f"{stats['avg_confidence']} avg confidence\n")
                break
            
            elif user_input.lower() == '/help':
                print(chatbot.show_help())
                continue
            
            elif user_input.lower() == '/stats':
                stats = chatbot.get_stats()
                print(f"\n📊 ML STATISTICS:")
                print(f"   Total Messages: {stats['total_messages']}")
                print(f"   Matched Responses: {stats['matched_responses']}")
                print(f"   Match Rate: {stats['match_rate']}")
                print(f"   Avg Confidence: {stats['avg_confidence']}")
                print(f"   Vocabulary Size: {stats['vocabulary_size']} terms")
                print(f"   Learned Responses: {stats['learned_responses']}")
                continue
            
            elif user_input.lower() == '/history':
                print(chatbot.show_history())
                continue
            
            elif user_input.lower() == '/clear':
                chatbot.conversation_history = []
                print("✨ Conversation history cleared!")
                continue
            
            elif user_input.lower() == '/about':
                print(chatbot.show_about())
                continue
            
            elif user_input.lower() == '/learn':
                print("\n🎓 LEARNING MODE")
                print("Enter the phrase/question you want me to learn:")
                learn_phrase = input("👤 Phrase: ").strip()
                if learn_phrase:
                    print(f"Now enter the response I should give for '{learn_phrase[:30]}...':")
                    learning_mode = True
                else:
                    print("❌ Learning cancelled - no phrase entered.")
                continue
            
            # Get response from chatbot with confidence
            response, confidence = chatbot.chat(user_input)
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            print(f"\n🤖 Bot: {response}")
            print(f"   📊 Confidence: [{confidence_bar}] {confidence:.2f}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
