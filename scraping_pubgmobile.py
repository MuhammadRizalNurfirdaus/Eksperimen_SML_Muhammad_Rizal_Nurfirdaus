from google_play_scraper import reviews, Sort
import pandas as pd

all_reviews = []
continuation_token = None

target_app_id = 'com.tencent.ig' 

print(f"Memulai scraping untuk: {target_app_id}")

while len(all_reviews) < 15000:
    result, continuation_token = reviews(
        target_app_id,   
        lang='id',       
        country='id',    
        sort=Sort.NEWEST,
        count=min(1000, 15000 - len(all_reviews)), 
        continuation_token=continuation_token
    )
    
    if not result:
        break
        
    all_reviews.extend(result)
    print(f"Sudah mengambil {len(all_reviews)} review...")

# Memproses data ke DataFrame
data = []
for r in all_reviews:
    data.append({
        'review': r['content'],
        'rating': r['score'],
        'date': r['at'],
        'userName': r['userName']
    })

df = pd.DataFrame(data)
df.to_csv('pubg_mobile_reviews.csv', index=False)

print("Scraping selesai! Data tersimpan di pubg_mobile_reviews.csv")