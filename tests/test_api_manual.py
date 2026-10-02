import requests


URL = (
    "http://127.0.0.1:8000"
    "/forecast"
)


payload = {

    "store_nbr": 10,

    "family":
        "GROCERY I",

    "onpromotion": 20,
}


response = requests.post(
    URL,
    json=payload,
)


print(
    "Status code:"
)

print(
    response.status_code
)


print(
    "\nResponse:"
)

print(
    response.json()
)