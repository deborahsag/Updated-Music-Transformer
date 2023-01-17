import requests
import re
import urllib.request
from tqdm import tqdm

urls = open("/mnt/e/Datasets/touhou_urls.txt", "r").readlines()
urls = [x[:-1] for x in urls]
for url in tqdm(urls):
    try:
        midiurl = re.search(r"href=\"https://upload.thwiki.cc.+\.midi?\">", requests.get(url).text).group()[6:-2]
    except Exception as e:
        print(f"Error with {url}: {e}")
    filename = midiurl[midiurl.rfind("/")+1:]
    urllib.request.urlretrieve(midiurl, f"/mnt/e/Datasets/touhou-midis/{filename}")
print("Done.")
