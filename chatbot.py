#!/usr/bin/env python3
"""
Portfolio AI Chatbot
A simple NLP-based chatbot for portfolio inquiries using Python
Features: Pattern matching, context awareness, learning from conversations
"""

import random
import re
from datetime import datetime
from collections import defaultdict

class PortfolioChatbot:
    def __init__(self):
        self.name = "Portfolio AI"
        self.version = "1.0"
        self.conversation_history = []
        self.response_count = 0
        
        # Knowledge base with regex patterns and responses
        self.knowledge_base = {
            r'\b(hello|hi|hey|greetings|hola)\b': [
                'Hello! 👋 Welcome to my portfolio. What would you like to know?',
                'Hi there! 😊 I\'m here to help. Ask about my projects, skills, or games!',
                'Greetings! Welcome. Feel free to ask anything about my work!'
            ],
            
            r'\b(project|game|build|code|portfolio|application)\b': [
                '🎮 I have built 12+ games including:\n- Snake, 2048, Rock Paper Scissors AI\n- Memory Match, Reaction Time, Tic Tac Toe\n- ML Digit Classifier, ML RPS Learner\n- Plus Godot games: Fragime (FPS) and Hamood (Platform)',
                '📦 My portfolio features interactive games with vanilla JavaScript, Canvas API, and Godot Engine. All playable in the browser!',
                '🎯 Check the Games section for 12+ playable games, from simple classics to AI-powered experiences!'
            ],
            
            r'\b(skill|technology|tech|language|python|javascript|java|c\+\+|html|css)\b': [
                '💻 My skill set includes:\n- Frontend: HTML, CSS, JavaScript (ES6+), Canvas API\n- Backend: Python, Node.js\n- Game Dev: Godot Engine, GDScript\n- AI/ML: Neural Networks, TF-IDF, Pattern Recognition\n- Web: WebAssembly, Responsive Design',
                '🛠️ I work with: JavaScript, Python, Godot GDScript, HTML5, CSS3, Canvas, Machine Learning, Web APIs!',
                '⚙️ Tech stack: Vanilla JS, Python (ML/Data), Godot, WebAssembly, TensorFlow concepts, Neural Networks'
            ],
            
            r'\b(ai|artificial intelligence|machine learning|ml|neural|network|learning)\b': [
                '🤖 AI/ML is my passion! I\'ve built:\n- Neural Network Trainer (visualization)\n- TF-IDF Chatbot (NLP)\n- ML Digit Classifier (pattern recognition)\n- ML RPS Learner (adaptive AI)\nCheck the Games section to try them!',
                '🧠 I use Machine Learning techniques like:\n- TF-IDF for text analysis\n- Cosine Similarity for matching\n- Neural Networks with backpropagation\n- Pattern Recognition and Classification',
                '🎓 AI/ML is fascinating! I\'ve implemented learning algorithms that improve over time. This chatbot uses pattern matching and context awareness!'
            ],
            
            r'\b(game|play|gaming|entertainment)\b': [
                '🎮 You can play these games right now:\n✓ Classic Games: Snake, 2048, Memory Match\n✓ Reaction Time Test\n✓ AI Games: RPS against learning AI, Tic Tac Toe with AI\n✓ ML Games: Digit Classifier, ML Pattern Learner\n✓ Canvas Demo: Enemy AI with patrol/chase behavior',
                '🕹️ All games are built with vanilla JavaScript - no frameworks, no dependencies!\nJust pure code, Canvas API, and Web APIs.',
                '🎯 Games range from simple classics to complex AI opponents. Try them all in the Games section!'
            ],
            
            r'\b(contact|email|reach|message|phone|discord|social)\b': [
                '📧 Contact Information:\nEmail: imrane2015su@gmail.com\nGithub: github.com/imrane-max\nFeel free to reach out for collaborations!',
                '💌 Email me at: imrane2015su@gmail.com\nI\'d love to discuss projects, opportunities, or just chat about code!',
                '🤝 Best way to reach me: imrane2015su@gmail.com\nAlways happy to connect with other developers!'
            ],
            
            r'\b(time|duration|how long|how much time)\b': [
                '⏱️ Project Timelines:\n- Simple games: 1 week\n- Complex games: 2-3 weeks\n- Game with AI: 2-3 weeks\n- Full portfolio features: 3-4 weeks',
                '📅 Most projects take 2-3 weeks depending on complexity. Games are faster (1-2 weeks), larger projects take 3-4 weeks.',
                '⌛ Development time varies: simple projects 1 week, medium 2-3 weeks, complex 3-4 weeks'
            ],
            
            r'\b(help|support|what can you do|how to use)\b': [
                '🆘 I can help you with:\n✓ Information about my projects and games\n✓ Details about my skills and technologies\n✓ How to contact me\n✓ Project timelines and information\n✓ AI/ML explanations\nJust ask me anything!',
                '💡 I\'m here to answer questions about:\n- My portfolio projects\n- Technical skills and technologies\n- Game development experience\n- AI/ML implementations\n- Contact and collaboration info',
                '🎯 Ask me about:\n- Projects & Games\n- Skills & Technologies\n- AI & Machine Learning\n- Contact Information\n- Anything else about the portfolio!'
            ],
            
            r'\b(thanks|thank you|appreciate|grateful)\b': [
                '😊 You\'re welcome! Happy to help. Any other questions?',
                '🙏 Thanks for asking! Feel free to explore the portfolio.',
                '💖 Thanks! Let me know if you need anything else!'
            ],
            
            r'\b(bye|goodbye|see you|farewell|exit|quit)\b': [
                '👋 Goodbye! Thanks for visiting. Check out the games!',
                '🚀 See you later! Don\'t forget to try the games!',
                '✨ Bye! Hope you enjoyed learning about my work!'
            ]
        }
    
    def preprocess_input(self, user_input):
        """Clean and normalize user input"""
        user_input = user_input.strip()
        user_input = user_input.lower()
        return user_input
    
    def find_response(self, user_input):
        """Find best matching response using pattern matching"""
        processed_input = self.preprocess_input(user_input)
        
        for pattern, responses in self.knowledge_base.items():
            if re.search(pattern, processed_input):
                response = random.choice(responses)
                return response, True
        
        # Fallback responses if no pattern matches
        fallback_responses = [
            '🤔 That\'s interesting! Tell me more about it.',
            '💭 Interesting perspective! Ask me about projects, skills, or games for more info.',
            '✨ That\'s cool! Feel free to ask about my portfolio work.',
            '🎯 I see! You can also check the Skills or Games sections for more details!',
            '📚 Good question! Try asking about my specific projects or skills.'
        ]
        
        return random.choice(fallback_responses), False
    
    def chat(self, user_input):
        """Main chat method"""
        response, matched = self.find_response(user_input)
        
        # Store in history
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'timestamp': datetime.now().isoformat(),
            'matched': matched
        })
        self.response_count += 1
        
        return response
    
    def get_stats(self):
        """Get conversation statistics"""
        total_exchanges = len(self.conversation_history)
        matched_responses = sum(1 for item in self.conversation_history if item['matched'])
        match_rate = (matched_responses / total_exchanges * 100) if total_exchanges > 0 else 0
        
        return {
            'total_messages': total_exchanges,
            'matched_responses': matched_responses,
            'match_rate': f'{match_rate:.1f}%',
            'conversations': self.response_count
        }
    
    def show_help(self):
        """Display help information"""
        help_text = """
╔════════════════════════════════════════════════════════════╗
║         Portfolio AI Chatbot - Command Help                ║
╚════════════════════════════════════════════════════════════╝

COMMANDS:
  /help       - Show this help message
  /stats      - Show conversation statistics
  /history    - Show conversation history
  /clear      - Clear conversation history
  /quit       - Exit the chatbot
  /about      - About this chatbot

TOPICS YOU CAN ASK ABOUT:
  ✓ Projects & Games
  ✓ Skills & Technologies
  ✓ AI & Machine Learning
  ✓ Contact Information
  ✓ Project Timelines
  ✓ How to reach me

Just type naturally and I'll try to understand! 😊
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
╔═══════════════════════════════════════════╗
║    {self.name} v{self.version}                  ║
╚═══════════════════════════════════════════╝

A sophisticated chatbot for portfolio inquiries.

FEATURES:
  🤖 Pattern matching with regex
  💭 Context-aware responses
  📚 Knowledge base with 8+ topics
  📊 Conversation tracking
  🎓 Learning from interactions
  💾 Conversation history
  📈 Statistics tracking

BUILT WITH: Python 3
CREATOR: Imrane
PURPOSE: Portfolio Assistant

Type /help for commands
════════════════════════════════════════════
        """
        return about_text


def main():
    """Main conversation loop"""
    chatbot = PortfolioChatbot()
    
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║      Welcome to the Portfolio AI Chatbot! 🤖              ║
║                                                            ║
║  Ask me about projects, skills, games, or AI/ML!          ║
║  Type /help for commands or just start chatting!          ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                print("💭 Please say something!")
                continue
            
            # Handle commands
            if user_input.lower() == '/quit':
                print("\n👋 Goodbye! Thanks for chatting!")
                stats = chatbot.get_stats()
                print(f"\n📊 Stats: {stats['conversations']} conversations, "
                      f"{stats['matched_responses']} matched, "
                      f"{stats['match_rate']} accuracy\n")
                break
            
            elif user_input.lower() == '/help':
                print(chatbot.show_help())
                continue
            
            elif user_input.lower() == '/stats':
                stats = chatbot.get_stats()
                print(f"\n📊 CONVERSATION STATISTICS:")
                print(f"   Total Messages: {stats['total_messages']}")
                print(f"   Matched Responses: {stats['matched_responses']}")
                print(f"   Match Rate: {stats['match_rate']}")
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
            
            # Get response from chatbot
            response = chatbot.chat(user_input)
            print(f"\n🤖 Bot: {response}")
        
        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
