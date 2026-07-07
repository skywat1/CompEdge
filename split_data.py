import pandas as pd

df = pd.read_csv('data/cleaned_sold.csv')

# Photos
df_images = df[['main_image-src', 'all_images', 'zpid']]
df_images = df_images.dropna()
df_images.to_csv('data/image_links.csv', index=False)