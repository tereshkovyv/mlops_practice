import os
import requests

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def download_data():
    os.makedirs("lab2/data/raw", exist_ok=True)
    response = requests.get(URL)
    with open("lab2/data/raw/titanic.csv", "wb") as f:
        f.write(response.content)

if __name__ == "__main__":
    download_data()