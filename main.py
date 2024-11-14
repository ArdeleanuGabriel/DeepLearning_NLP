import requests

url = "https://libretranslate.com/translate"

# Multipart form data as per the request details
multipart_form_data = {
    'q': 'Nu stiu ce sa pun aici',
    'source': 'auto',
    'target': 'en',
    'format': 'text',
    'alternatives': '3',
    'api_key': ''
}

# Headers to match the original request
headers = {
    "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundaryaHehRPcYadQuHE6z",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
    "Origin": "https://libretranslate.com",
    "Referer": "https://libretranslate.com/?source=auto&target=en&q=Nu+stiu+ce+sa+pun+aici",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": "r=1; session=d6d12e53-f217-4e41-af71-31017f6db6f0; _ga=GA1.1.918427248.1729762719; _ga_KPKM1EP5EW=GS1.1.1729762718.1.1.1729763052.0.0.0"
}

# Send the POST request
response = requests.post(url, files=multipart_form_data, headers=headers)

# Check the response
print(f"Status Code: {response.status_code}")
print(f"Response: {response.text}")
