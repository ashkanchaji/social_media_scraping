import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

# تنظیمات
CHANNEL = 'bourse24ir'  # نام کاربری کانال بورسی
ONE_MONTH_AGO = datetime.now() - timedelta(days=30)
BASE_URL = f"https://t.me/s/{CHANNEL}"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'
}

messages = []
before_id = ""

while True:
    url = f"{BASE_URL}?before={before_id}" if before_id else BASE_URL
    res = requests.get(url, headers=headers)
    
    if res.status_code != 200:
        break

    soup = BeautifulSoup(res.text, 'html.parser')
    posts = soup.find_all('div', class_='tgme_widget_message')
    
    if not posts:
        break

    stop_scraping = False
    
    for post in reversed(posts):
        # دریافت شناسه پیام
        post_id = post.get('data-post', '').split('/')[-1]
        
        # دریافت تاریخ پیام
        time_tag = post.find('time', class_='time')
        if not time_tag or not time_tag.has_attr('datetime'):
            continue
            
        post_date = datetime.fromisoformat(time_tag['datetime'].replace('Z', '+00:00')).replace(tzinfo=None)
        
        # اگر پیام قدیمی‌تر از ۳۰ روز گذشته بود، متوقف شود
        if post_date < ONE_MONTH_AGO:
            stop_scraping = True
            break
            
        # دریافت متن پیام
        text_tag = post.find('div', class_='tgme_widget_message_text')
        text = text_tag.get_text(separator="\n", strip=True) if text_tag else ""
        
        if text:
            messages.append({
                'id': post_id,
                'date': str(post_date),
                'text': text
            })

    if stop_scraping:
        break

    # دریافت شناسه اولین پست صفحه برای بارگذاری پیام‌های قبلی
    first_post = posts[0].get('data-post', '')
    if first_post:
        before_id = first_post.split('/')[-1]
    else:
        break

# چاپ تعداد و نمونه پیام‌ها
print(f"تعداد پیام‌های استخراج‌شده در ۳۰ روز اخیر: {len(messages)}")
for m in messages[:3]:
    print(f"\n--- [{m['date']}] ---\n{m['text'][:150]}...")