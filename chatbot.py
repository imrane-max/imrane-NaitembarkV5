#!/usr/bin/env python3
"""
Portfolio AI Chatbot with Machine Learning
An NLP-based chatbot using TF-IDF and Cosine Similarity for intelligent responses
Features: TF-IDF vectorization, cosine similarity matching, learning from conversations
"""

import random
import re
import math
from datetime import datetime
from collections import defaultdict, Counter


class TFIDF:
    """TF-IDF Vectorizer for text similarity"""
    
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.idf = {}
        self.doc_vectors = []
    
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
    
    def fit(self, documents):
        """Build vocabulary and compute IDF values"""
        self.documents = documents
        doc_count = len(documents)
        term_doc_freq = defaultdict(int)
        
        # Build vocabulary and document frequency
        for doc in documents:
            tokens = set(self.tokenize(doc))
            self.vocabulary.update(tokens)
            for token in tokens:
                term_doc_freq[token] += 1
        
        # Compute IDF: log(N / df) + 1 (smoothed)
        for term in self.vocabulary:
            self.idf[term] = math.log(doc_count / (term_doc_freq[term] + 1)) + 1
        
        # Compute document vectors
        self.doc_vectors = [self._compute_tfidf(doc) for doc in documents]
    
    def _compute_tfidf(self, text):
        """Compute TF-IDF vector for a text"""
        tokens = self.tokenize(text)
        if not tokens:
            return {}
        
        # Term frequency (normalized)
        tf = Counter(tokens)
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
    
    def find_most_similar(self, query, threshold=0.1):
        """Find the most similar document to the query"""
        query_vector = self._compute_tfidf(query)
        
        best_idx = -1
        best_score = threshold
        
        for idx, doc_vector in enumerate(self.doc_vectors):
            score = self.cosine_similarity(query_vector, doc_vector)
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx, best_score


class PortfolioChatbot:
    def __init__(self):
        self.name = "Portfolio ML AI"
        self.version = "2.0"
        self.conversation_history = []
        self.response_count = 0
        self.learned_responses = {}  # For learning from conversations
        
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
        """Find best matching response using TF-IDF ML matching"""
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
            '🎯 I use TF-IDF matching - try keywords like "skills", "games", "AI", or "contact"!'
        ]
        
        return random.choice(fallback_responses), 0.0, False
    
    def chat(self, user_input):
        """Main chat method with ML matching"""
        response, confidence, matched = self.find_response(user_input)
        
        # Store in history with confidence score
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat(),
            'matched': matched,
            'confidence': confidence
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
  🧠 TF-IDF Vectorization
  📊 Cosine Similarity Matching
  🎓 Learning from conversations
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

ML ALGORITHM:
  🧠 TF-IDF (Term Frequency-Inverse Document Frequency)
  📊 Cosine Similarity for semantic matching
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
    """Main conversation loop with ML chatbot"""
    chatbot = PortfolioChatbot()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║   Welcome to the Portfolio ML AI Chatbot! 🤖🧠            ║
║                                                            ║
║   Powered by TF-IDF & Cosine Similarity                   ║
║   Ask me about projects, skills, games, or AI/ML!         ║
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
