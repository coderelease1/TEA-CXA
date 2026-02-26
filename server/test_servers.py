import requests
import os

SERVERS_HOST = os.getenv("SERVERS_HOST", "dgx-039")#"127.0.0.1"



#-----------------------------------------medgemma-------------------------------------------

question = "Question: Is this image normal?\nA) no\nB) yes"
image_path_list = ["/your/path/cheXbench/slake/imgs/xmlab309/source.jpg"]


request_data = {
    "image_path": image_path_list[0],
    "prompt": question,
    #"max_new_tokens": 300
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

for port in ["5007","5009","5010","5011"]:
    response = requests.post(
        f"http://{SERVERS_HOST}:{port}/medgemma",
        json=request_data,
        headers=headers,
        proxies=proxies,
        timeout=50
    )
    response.raise_for_status()

    result = response.json()
    print(result)


#-----------------------------------------lingshu-------------------------------------------

question = "Question: Is this image normal?\nA) no\nB) yes"
image_path_list = ["/your/path/cheXbench/slake/imgs/xmlab309/source.jpg"]

request_data = {
    "image_path": image_path_list[0],
    "prompt": question,
    #"max_new_tokens": 300
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

for port in ["5006", "5012"]:
    response = requests.post(
        f"http://{SERVERS_HOST}:{port}/lingshu",
        json=request_data,
        headers=headers,
        proxies=proxies,
        timeout=50
    )
    response.raise_for_status()

    result = response.json()
    print(result)