import requests
from bs4 import BeautifulSoup
import json

# Hedef URL
url = 'https://seytim.org/index.php'

# Web sayfasının HTML içeriğini çek
response = requests.get(url)

# HTML içeriğini BeautifulSoup ile parse et
soup = BeautifulSoup(response.content, 'html.parser')

# "cbox-4-txt" sınıfına sahip öğeleri bul
courses_grid = soup.find_all('div', class_='cbox-4-txt')

# Eğitimler listesi
educations = []

# Bulunan öğeleri işleyip eğitimler listesine ekle
for course in courses_grid:
    # Başlık ve tarih bilgilerini ayıklayın
    title = course.find('h5').text.strip() if course.find('h5') else "Başlık bulunamadı"
    date = course.find('p').text.strip() if course.find('p') else "Tarih bulunamadı"
    
    # Eğitim bilgilerini listeye ekleyin
    educations.append({
        'title': title,
        'date': date
    })

# Eğitimleri tek bir responses alanında birleştir
response_text = "\n".join([f"{edu['title']} kursu {edu['date']} tarihlerinde gerçekleştirilecektir." for edu in educations])

# Yeni intents formatında hazırla
new_intents = [
    {
        "tag": "egitim",
        "patterns": ["egitim"],
        "responses": [response_text]
    }
]

# Mevcut intents dosyasını oku
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Yeni kurs bilgilerini intents dosyasına ekle
data['intents'].extend(new_intents)

# Güncellenmiş intents dosyasını yaz
with open('intents.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Kurs bilgileri intents dosyasına eklendi.")
