import requests
from mcp.server.fastmcp import FastMCP
from typing import List

import os
import threading

import jsonlines
import hashlib
from typing import Dict, Any

def read_server_host_from_file(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            return s
    

SERVERS_HOST = os.getenv("SERVERS_HOST", "127.0.0.1")
print(SERVERS_HOST)
SERVERS_HOST = read_server_host_from_file("/your/path/to/servers_node.txt")

mcp = FastMCP("LocalServer")

#@mcp.tool()
def classification(image_path_list, img_id: int):
    """
    Tool that classifies a chest X-ray image for multiple pathologies.
    This tool uses a pre-trained DenseNet model to analyze a chest X-ray image and predict the likelihood of various pathologies. The model can classify the following 18 conditions:
    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Enlarged Cardiomediastinum, Fibrosis, Fracture, Hernia, Infiltration, Lung Lesion, Lung Opacity, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax
    The output values represent the probability (from 0 to 1) of each condition being present in the image.
    A higher value indicates a higher likelihood of the condition being present.

    Args:
        img_id (int): The index of the figure in this query (1,2,3...).

    Returns:
        str: JSON string of a dictionary of pathologies and their predicted probabilities (0 to 1).
    """
    if img_id < 1 or img_id > len(image_path_list):
        return "⚠️ img_id out of range!"
    try:
        request_data = {
            "image_path": image_path_list[img_id-1]
        }
        headers = {
            "Content-Type": "application/json"
        }
        proxies = {
            "http": None,
            "https": None
        }

        response = requests.post(
            f"http://{SERVERS_HOST}:5003/classification",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        # Normalize minimal validation

        return result

    except requests.exceptions.Timeout:
        return "⚠️ Classification service request timeout, please check if the service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to classification service, please ensure that the service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Classification service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ Classification failed: {str(e)}\nError type: {type(e).__name__}"

#@mcp.tool()
def vqa_old(image_path_list, img_ids: List[int], prompt: str):
    """
    Tool that performs chest X-ray Visual Question Answering (VQA) using CheXagent.
    Can perform multiple tasks including: visual question answering, report generation, abnormality detection, comparative analysis, anatomical description, and clinical interpretation.
    
    Args:
        img_ids (List[int]): The list of indices (1,2,3...) of X-ray figures in this query.
        prompt (str): The question or instruction.

    Returns:
        The VQA response
    """
    if not isinstance(img_ids, list):
        return "⚠️ img_ids should be a list of indices of X-ray figures in this query!"
    for id in img_ids:
        if id < 1 or id > len(image_path_list):
            return "⚠️ img_id out of range!"
    try:
        headers = {"Content-Type": "application/json"}
        proxies = {"http": None, "https": None}
        payload = {
            "image_paths": [image_path_list[id-1] for id in img_ids],
            "prompt": prompt,
        }
        response = requests.post(
            f"http://{SERVERS_HOST}:5004/vqa",
            json=payload,
            headers=headers,
            proxies=proxies,
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return "⚠️ VQA service request timeout, please check if the VQA service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to VQA service, please ensure that the VQA service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ VQA service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ VQA failed: {str(e)}\nError type: {type(e).__name__}"

#@mcp.tool()
def report_generation_old(image_path_list, img_id: int):
    """
    Tool that generates a comprehensive chest X-ray report (Findings + Impression) using two ViT-BERT models.

    Args:
        img_id (int): The index of the figure in this query (1,2,3...).

    Returns:
        JSON dict of the report.
    """
    if img_id < 1 or img_id > len(image_path_list):
        return "⚠️ img_id out of range!"
    try:
        request_data = {
            "image_path": image_path_list[img_id-1]
        }
        headers = {
            "Content-Type": "application/json"
        }
        proxies = {
            "http": None,
            "https": None
        }

        response = requests.post(
            f"http://{SERVERS_HOST}:5005/report_generation",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        return result

    except requests.exceptions.Timeout:
        return "⚠️ Report generation service request timeout, please check if the service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to report generation service, please ensure that the service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Report generation service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}\nError type: {type(e).__name__}"

#PG_PORTS = ["5006", "5012", "5013", "5014"]
PG_PORTS = ["5006", "5012"]
_pg_rr_index = 0
_pg_rr_lock = threading.Lock()

def get_next_pg_port() -> str:
    global _pg_rr_index
    with _pg_rr_lock:
        port = PG_PORTS[_pg_rr_index % len(PG_PORTS)]
        _pg_rr_index += 1
        return port

        
#@mcp.tool()
def phrase_grounding(image_path_list, img_id: int, phrase: str, extra_info):
    """
    Tool that grounds a medical phrase in a frontal chest X-ray image.
    It returns predicted bounding box coordinates in format [x_topleft, y_topleft, x_bottomright, y_bottomright], where each value is between 0-1 representing relative position in the image.

    Args:
        img_id (int): The index (1,2,3...) of the figure (must be a *frontal* chest X-ray image) in this query.
        phrase (str): Medical phrase to locate (e.g., 'Pleural effusion', 'Cardiomegaly').

    Returns:
        JSON dict with predictions/metadata.
    """
    if img_id < 1 or img_id > len(image_path_list):
        return "⚠️ img_id out of range!"
    if not isinstance(phrase, str) or len(phrase.strip()) == 0:
        return "⚠️ phrase must be a non-empty string."
    if extra_info["ImageViewPosition"][img_id-1] == "LATERAL":
        return "⚠️ This tool can only accept a *frontal* chest X-ray image!"
    try:
        request_data = {
            "image_path": image_path_list[img_id-1],
            "phrase": phrase,
        }
        headers = {
            "Content-Type": "application/json"
        }
        proxies = {
            "http": None,
            "https": None
        }

        pg_port = get_next_pg_port()
        response = requests.post(
            f"http://{SERVERS_HOST}:{pg_port}/phrase_grounding",
            json=request_data,
            headers=headers,
            proxies=proxies,
            timeout=60  
        )
        response.raise_for_status()

        result = response.json()
        return result

    except requests.exceptions.Timeout:
        return "⚠️ Phrase grounding service request timeout, please check if the service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to phrase grounding service, please ensure that the service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Phrase grounding service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ Phrase grounding failed: {str(e)}\nError type: {type(e).__name__}"


VQA_PORTS = ["5007","5009","5010","5011"]#["5007","5009","5010"]

# 轮询计数器与锁（线程安全）
_vqa_rr_index = 0
_vqa_rr_lock = threading.Lock()

def get_next_port() -> str:
    global _vqa_rr_index
    with _vqa_rr_lock:
        port = VQA_PORTS[_vqa_rr_index % len(VQA_PORTS)]
        _vqa_rr_index += 1
        return port
    
#######
# original place for vqa tools
    
#@mcp.tool()
def report_generation_server(
    image_path_list,
    #context,
    img_ids: List[int],
    extra_info
):
    """
    Tool that generates a comprehensive chest X-ray report (Findings + Impression).
    This tool can provide valuable and complementary information for answering a question.

    Args:
        img_ids (List[int]): The list of indices (1,2,3...) of all the X-ray figures in this query (1-based). The indices of all the figures should be given.
        
    Returns:
        JSON dict of the report.
    """
    if not isinstance(img_ids, list):
        return "⚠️ img_ids should be a list of indices of X-ray figures in this query!"
    if len(image_path_list) == 0:
        return "⚠️ image_path_list is empty!"
    for id in img_ids:
        if not isinstance(id, int):
            return "⚠️ img_ids must contain integers!"
        if id < 1 or id > len(image_path_list):
            return "⚠️ img_id out of range!"
    default_img_ids = list(range(1, len(image_path_list)+1))
    if img_ids != default_img_ids:
        return "⚠️ img_ids should be the list of indices of all the X-ray figures in this query!"

    try:
        headers = {"Content-Type": "application/json"}
        proxies = {"http": None, "https": None}
        payload = {
            "image_paths": [image_path_list[id - 1] for id in img_ids],
            "context": extra_info["context"]#context,
        }

        response = requests.post(
            f"http://{SERVERS_HOST}:5008/report_generation",
            json=payload,
            headers=headers,
            proxies=proxies,
            timeout=80,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return "⚠️ Report generation service request timeout, please check if the report generation service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to report generation service, please ensure that the report generation service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Report generation service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ Report generation failed: {str(e)}\nError type: {type(e).__name__}"

MEDVERSA_CACHE_PATH = "/your/path/server/medversa_cache_chexbench.jsonl"

def make_key(image_paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in image_paths:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

_report_cache_index: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

""" with _cache_lock:
    with jsonlines.open(MEDVERSA_CACHE_PATH, "r") as reader:
        for obj in reader:
            k = obj.get("key")
            _report_cache_index[k] = obj """

#@mcp.tool()
def report_generation(
    image_path_list,
    #context,
    img_ids: List[int],
    extra_info
):
    """
    Tool that generates a comprehensive chest X-ray report (Findings + Impression).
    This tool can provide very valuable and useful information for answering a question.

    Args:
        img_ids (List[int]): The list of indices (1,2,3...) of *all* the X-ray figures in this query (1-based). The indices of all the figures should be given.
        
    Returns:
        JSON dict of the report.
    """
    if not isinstance(img_ids, list):
        return "⚠️ img_ids should be a list of indices of X-ray figures in this query!"
    if len(image_path_list) == 0:
        return "⚠️ image_path_list is empty!"
    for id in img_ids:
        if not isinstance(id, int):
            return "⚠️ img_ids must contain integers!"
        if id < 1 or id > len(image_path_list):
            return "⚠️ img_id out of range!"
    default_img_ids = list(range(1, len(image_path_list)+1))
    if img_ids != default_img_ids:
        #return "⚠️ img_ids should be the list of indices of all the X-ray figures in this query!"
        pre_str = "⚠️ img_ids can only be the list of indices of *all* the X-ray figures in this query! However this time I still give you the report for all the X-ray figures in this query.\n"
    else:
        pre_str = ""

    image_paths = image_path_list
    key = make_key(image_paths)##[p.replace('/your/path/','') for p in image_paths]

    cached = _report_cache_index.get(key)
    if cached is not None:
        return pre_str + str(cached.get("report"))
    else:
        with open('/your/path/error.txt', 'a', encoding='utf-8') as file:
            file.write(f"cannot find report for {image_path_list}\n")

@mcp.tool()
def vqa_medgemma(image_path_list, img_id: int, prompt: str, extra_info):
    """
    Tool that performs chest X-ray Visual Question Answering (VQA) using the medgemma model.
    Supports only single-image queries—choose one image to use.
    
    Args:
        img_id (int): The index of the figure in this query (1,2,3...).
        prompt (str): A question and explicit answer options (separated by newline character or comma).

    Returns:
        The VQA response
    """
    if img_id < 1 or img_id > len(image_path_list):
        return "⚠️ img_id out of range!"
    try:
        headers = {"Content-Type": "application/json"}
        proxies = {"http": None, "https": None}
        payload = {
            "image_path": image_path_list[img_id - 1],
            #"prompt": "Context: " + extra_info["context"] + "\n" + "Question: " + prompt,
            "prompt": "Question: " + prompt,
        }
        vqa_port = get_next_port()
        response = requests.post(
            f"http://{SERVERS_HOST}:{vqa_port}/medgemma",#5007
            json=payload,
            headers=headers,
            proxies=proxies,
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return "⚠️ medgemma service request timeout, please check if the VQA service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to medgemma service, please ensure that the VQA service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ medgemma service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ medgemma failed: {str(e)}\nError type: {type(e).__name__}"

@mcp.tool()
def vqa_lingshu(image_path_list, img_id: int, prompt: str, extra_info):
    """
    Tool that performs chest X-ray Visual Question Answering (VQA) using the lingshu model.
    Supports only single-image queries—choose one image to use.
    
    Args:
        img_id (int): The index of the figure in this query (1,2,3...).
        prompt (str): A question and explicit answer options (separated by newline character or comma).

    Returns:
        The VQA response
    """
    if img_id < 1 or img_id > len(image_path_list):
        return "⚠️ img_id out of range!"
    try:
        headers = {"Content-Type": "application/json"}
        proxies = {"http": None, "https": None}
        payload = {
            "image_path": image_path_list[img_id - 1],
            #"prompt": "Context: " + extra_info["context"] + "\n" + "Question: " + prompt,
            "prompt": "Question: " + prompt,
        }
        vqa2_port = get_next_pg_port()
        response = requests.post(
            f"http://{SERVERS_HOST}:{vqa2_port}/lingshu",
            json=payload,
            headers=headers,
            proxies=proxies,
            timeout=180
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return "⚠️ lingshu service request timeout, please check if the VQA service is running properly."
    except requests.exceptions.ConnectionError:
        return "⚠️ Unable to connect to lingshu service, please ensure that the VQA service is running."
    except requests.exceptions.RequestException as e:
        return f"⚠️ lingshu service request failed: {str(e)}\nDetail: {e.response.text if hasattr(e, 'response') and e.response is not None else 'No detail'}"
    except Exception as e:
        return f"⚠️ lingshu failed: {str(e)}\nError type: {type(e).__name__}"
        

if __name__ == "__main__":
    print("\nStart MCP service:")
    mcp.run(transport='stdio')