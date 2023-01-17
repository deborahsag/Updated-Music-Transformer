import requests
import re
import urllib.request
from tqdm import tqdm

midis = open("/mnt/f/Datasets/vgmusic_snes_urls.txt", "r").readlines()
midis = [x[:-1] for x in midis]
url = "http://vgmusic.com/music/console/nintendo/snes/"
for midi in tqdm(midis):
    urllib.request.urlretrieve(f"{url}{midi}", f"/mnt/f/Datasets/vgmusic-snes/{midi}")
print("Done.")
