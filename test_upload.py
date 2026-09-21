import requests

url = "http://127.0.0.1:8000/health"

print("Sending request...")
response = requests.get(url, timeout=10)

print("Status code:", response.status_code)
print("Response:", response.text)
print("Test completed.")