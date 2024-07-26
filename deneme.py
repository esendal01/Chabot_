import sys
import random
import json
import numpy as np
import torch
import re
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel, QScrollArea, QFrame, QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Model yükleme
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "SEY-Bot"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    return "I do not understand..."

class ChatBotUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ChatBot")
        self.setGeometry(100, 100, 600, 800)  # Pencereyi genişletme

        self.layout = QVBoxLayout()

        # Scroll area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setSpacing(10)  # Mesaj kutuları arasındaki boşluğu ayarlama
        self.scroll.setWidget(self.scroll_content)

        # Cevap veriliyor mesajı
        self.typing_label = QLabel()
        self.typing_label.setStyleSheet("color: white; font-weight: bold; padding: 10px; background-color: rgba(44, 62, 80, 150);")  # Daha açık lacivert arka plan
        self.typing_label.setAlignment(Qt.AlignCenter)
        self.typing_label.setText("Bot is typing...")
        self.typing_label.hide()

        # Butonlar için yatay düzen
        self.buttons_layout = QHBoxLayout()
        self.wifi_button = QPushButton("Wifi")
        self.wifi_button.setStyleSheet("background-color: #64B5F6; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.wifi_button.clicked.connect(self.handle_wifi)
        self.buttons_layout.addWidget(self.wifi_button)

        self.egitim_button = QPushButton("Eğitim")
        self.egitim_button.setStyleSheet("background-color: #64B5F6; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.egitim_button.clicked.connect(self.handle_egitim)
        self.buttons_layout.addWidget(self.egitim_button)

        # Kullanıcı girdisi için düzenleyici
        self.input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.setStyleSheet("background-color: white; border: 1px solid #ccc; padding: 10px; border-radius: 5px;")
        self.input_layout.addWidget(self.user_input)
        
        # Gönder butonu
        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet("background-color: #64B5F6; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.send_button.clicked.connect(self.send_message)
        self.input_layout.addWidget(self.send_button)

        # Layout'ları ayarla
        self.layout.addWidget(self.scroll)
        self.layout.addWidget(self.typing_label)  # typing_label butonlardan önce eklenir
        self.layout.addLayout(self.buttons_layout)
        self.layout.addLayout(self.input_layout)
        self.setLayout(self.layout)

        # Arka planı beyaz yapma
        self.setStyleSheet("background-color: white;")

        # Timer for typing animation
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self.update_typing_animation)
        self.typing_dots = 0

        # İlk mesajı gönder
        QTimer.singleShot(1000, self.send_initial_message)  # İlk mesajı 1 saniye sonra gönder

    def send_initial_message(self):
        initial_message = "Size nasıl yardımcı olabilirim?"
        self.add_message(f"Bot: {initial_message}", user=False)
        self.buttons_layout.setEnabled(True)  # Butonları aktif et

    def handle_wifi(self):
        self.add_message("You: Wifi", user=True)
        # Bot yazıyor animasyonunu göster
        self.typing_label.show()
        self.typing_dots = 0
        self.typing_timer.start(500)  # Start typing animation every 500 ms
        
        # Gecikmeli yanıt
        QTimer.singleShot(3000, self.show_wifi_response)  # 3000 ms (3 saniye) gecikme

    def handle_egitim(self):
        self.add_message("You: Eğitim", user=True)
        # Bot yazıyor animasyonunu göster
        self.typing_label.show()
        self.typing_dots = 0
        self.typing_timer.start(500)  # Start typing animation every 500 ms

        # Gecikmeli yanıt
        QTimer.singleShot(3000, self.show_egitim_response)  # 3000 ms (3 saniye) gecikme

    def show_wifi_response(self):
        response = next(intent['responses'] for intent in intents['intents'] if intent['tag'] == "items")[0]
        self.add_message(f"Bot: {response}", user=False)
        self.typing_timer.stop()  # Stop typing animation timer
        self.typing_label.hide()  # Yazıyor animasyonunu gizle

    def show_egitim_response(self):
        response = next(intent['responses'] for intent in intents['intents'] if intent['tag'] == "egitim")[0]
        self.add_message(f"Bot: {response}", user=False)
        self.typing_timer.stop()  # Stop typing animation timer
        self.typing_label.hide()  # Yazıyor animasyonunu gizle

    def send_message(self):
        user_text = self.user_input.text()
        if user_text.strip() != "":
            self.add_message(f"You: {user_text}", user=True)
            
            # Bot yazıyor animasyonunu göster
            self.typing_label.show()
            self.typing_dots = 0
            self.typing_timer.start(500)  # Start typing animation every 500 ms
            
            # Gecikmeli yanıt
            QTimer.singleShot(3000, lambda: self.show_bot_response(user_text))  # 3000 ms (3 saniye) gecikme

    def update_typing_animation(self):
        if self.typing_dots >= 3:
            self.typing_dots = 0
        self.typing_label.setText("Bot is typing" + "." * self.typing_dots)
        self.typing_dots += 1

    def show_bot_response(self, user_text):
        response = get_response(user_text)
        self.add_message(f"Bot: {response}", user=False)
        self.user_input.clear()
        self.typing_timer.stop()  # Stop typing animation timer
        self.typing_label.hide()  # Yazıyor animasyonunu gizle

    def add_message(self, message, user=True):
        message_label = QLabel()
        message_label.setWordWrap(True)
        message_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Dinamik boyutlandırma
        message_label.setMinimumWidth(100)  # Minimum genişlik belirleme, çok küçük olmasını engeller
        message_label.setMaximumWidth(500)  # Maksimum genişlik belirleme
        message_label.setText(message)  # Mesaj metnini ayarla

        if user:
            message_label.setStyleSheet("background-color: lightblue; padding: 10px; border-radius: 10px; margin: 5px;")
            message_label.setAlignment(Qt.AlignRight)
            message_label.setFixedHeight(min(message_label.sizeHint().height(), 100))  # Maksimum yükseklik ayarı
            # Kullanıcı mesajları için frame ve layout oluşturma
            message_frame = QFrame()
            message_layout = QVBoxLayout()
            message_layout.setContentsMargins(0, 0, 0, 0)
            message_layout.addWidget(message_label)
            message_frame.setLayout(message_layout)
            self.scroll_layout.addWidget(message_frame)
        else:
            # Mesaj içindeki linkleri HTML bağlantısı yapma
            html_message = self.format_links(message)
            message_label.setText(html_message)
            message_label.setOpenExternalLinks(True)  # Bağlantıları tıklanabilir yapar
            message_label.setStyleSheet("background-color: darkblue; color: white; padding: 10px; border-radius: 10px; margin: 5px;")
            message_label.setAlignment(Qt.AlignLeft)
            message_label.setFixedHeight(min(message_label.sizeHint().height(), 500))  # Maksimum yükseklik ayarı

            # Mesaj ve resim için bir yatay düzen
            bot_message_layout = QHBoxLayout()
            bot_message_layout.addWidget(message_label)
            bot_message_layout.setContentsMargins(0, 0, 0, 0)

            # Yeni bir frame ve layout oluşturma
            message_frame = QFrame()
            message_frame.setLayout(bot_message_layout)
            self.scroll_layout.addWidget(message_frame)

        # Mesajın içeriğe göre genişleyip daralması
        message_label.adjustSize()
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def format_links(self, message):
        """
        Format message to make URLs clickable and handle line breaks
        """
        # Replace newlines with <br> tags
        message = message.replace('\n', '<br>')
        
        # Regex pattern to find URLs
        url_pattern = re.compile(r'(https?://[^\s]+)')
        # Replace URLs with HTML anchor tags
        formatted_message = url_pattern.sub(r'<a href="\1" style="color: #1E90FF; text-decoration: none;">\1</a>', message)
        return formatted_message

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ChatBotUI()
    main_window.show()
    sys.exit(app.exec_())
