import requests
url = "https://jsonplaceholder.typicode.com/comments/1"

response = requests.get(url, timeout=5)

if response.status_code == 200:
    data = response.json()
    print("Post Title:" ,data["email"])
else:
    print("Failed to fetch data. Status code:", response.status_code)