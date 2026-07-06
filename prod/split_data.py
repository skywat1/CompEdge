import pandas as pd

df = pd.read_csv('data/cleaned_sold.csv')

# Photos
df_photos = df[['main_image-src', 'all_images', 'zpid']]
df_photos = df_photos.dropna()
df_photos.to_csv('data/photo_links.csv', index=False)