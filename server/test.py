import requests

""" image_path_list = ["figures/4124/figure_1a.jpg", "figures/4124/figure_1b.jpg"]

request_data = {
    "image_path": image_path_list[0]
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

response = requests.post(
    "http://127.0.0.1:5003/classification",
    json=request_data,
    headers=headers,
    proxies=proxies,
    timeout=30
)
response.raise_for_status()

result = response.json()
print(result) """

#-----------------------------------------vqa-------------------------------------------
""" {"extra_info": {"answer": "C", "index": 3, "split": "train"}}

image_path_list = ["figures/1801/figure_1a.jpg", "figures/1801/figure_1b.jpg"]
question = 'Based on the provided chest X-ray images, which of the following findings most strongly supports the diagnosis of pectus excavatum, and what is the next appropriate step in managing this condition?\n\nA) The presence of sloping anterior ends of the ribs; monitor the patient for symptoms of respiratory distress.\nB) Apparent cardiomegaly with leftward shift of the heart; recommend immediate surgical intervention.\nC) Depression of the lower half of the sternum with decreased space between the sternum and dorsal spine; consider calculating the pectus index to evaluate the need for surgery.\nD) Ill-defined right heart border simulating right middle lobe consolidation; suggest a follow-up CT scan for further evaluation.\nE) Prominence of the main pulmonary artery with loss of the descending aortic interface; initiate treatment for suspected pulmonary hypertension.\nF) Good delineation of the lower dorsal spine through the heart; conduct a cardiac MRI to assess for underlying congenital heart disease.' """

#"reward_model": {"style": "rule", "ground_truth": "B"}

""" image_path_list = ["/your/path/datasets/deid_png_8bit_new/GRDNHQ3ABUU379NW/GRDNTWX0DY79N2T0/studies/1.2.826.0.1.3680043.8.498.43796459413844347731685797085847088019/series/1.2.826.0.1.3680043.8.498.11565728954623747828655037442713643529/instances/1.2.826.0.1.3680043.8.498.69192028213893186685576451535087979334.png"]
question = "What is the finding related to the marked site overlying the left lateral sixth rib on the chest X-ray?\nA) Displaced rib fracture\nB) Non-displaced rib fracture\nC) No radiographic abnormality\nD) Rib deformity"

request_data = {
    "image_paths": image_path_list,
    "prompt": question
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

response = requests.post(
    "http://dgx-039:5004/vqa",
    json=request_data,
    headers=headers,
    proxies=proxies,
    timeout=30
)
response.raise_for_status()

result = response.json()
print(result) """

#-----------------------------------------report_generation-------------------------------------------
""" image_path_list = ["figures/4124/figure_1a.jpg", "figures/4124/figure_1b.jpg"]

request_data = {
    "image_path": image_path_list[1]
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

response = requests.post(
    "http://127.0.0.1:5005/report_generation",
    json=request_data,
    headers=headers,
    proxies=proxies,
    timeout=120
)
response.raise_for_status()

result = response.json()
print(result) """

#-----------------------------------------phrase_grounding-------------------------------------------
#image_path_list = ["figures/4124/figure_1a.jpg", "figures/4124/figure_1b.jpg"]
""" image_path_list = ["/your/path/datasets/deid_png_8bit_new/GRDNKW0BUYTSL0MW/GRDNF3PVUNQHVBAZ/studies/1.2.826.0.1.3680043.8.498.52627801906649242602530898433256619302/series/1.2.826.0.1.3680043.8.498.68106848114609610539734175319479061803/instances/1.2.826.0.1.3680043.8.498.25183966107750743115028419519378386247.png", "/your/path/datasets/deid_png_8bit_new/GRDNKW0BUYTSL0MW/GRDNF3PVUNQHVBAZ/studies/1.2.826.0.1.3680043.8.498.52627801906649242602530898433256619302/series/1.2.826.0.1.3680043.8.498.74669647622394729407135645754726004648/instances/1.2.826.0.1.3680043.8.498.70554304080542314710274771981350556785.png", "/your/path/datasets/deid_png_8bit_new/GRDNKW0BUYTSL0MW/GRDNF3PVUNQHVBAZ/studies/1.2.826.0.1.3680043.8.498.52627801906649242602530898433256619302/series/1.2.826.0.1.3680043.8.498.86744969029283059756352630839195475573/instances/1.2.826.0.1.3680043.8.498.70667404585708557500451523874451962214.png"]
image_path = image_path_list[0]
phrase = 'Cardiomegaly'

request_data = {
    "image_path": image_path,
    "phrase": phrase,
    #"max_new_tokens": 500
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

response = requests.post(
    "http://dgx-039:5006/phrase_grounding",
    json=request_data,
    headers=headers,
    proxies=proxies,
    timeout=120
)
response.raise_for_status()

result = response.json()
print(result)
print(type(result)) """


#-----------------------------------------medgemma-------------------------------------------
""" {"extra_info": {"answer": "C", "index": 3, "split": "train"}}

image_path_list = ["figures/1801/figure_1a.jpg", "figures/1801/figure_1b.jpg"]
question = 'Based on the provided chest X-ray images, which of the following findings most strongly supports the diagnosis of pectus excavatum, and what is the next appropriate step in managing this condition?\n\nA) The presence of sloping anterior ends of the ribs; monitor the patient for symptoms of respiratory distress.\nB) Apparent cardiomegaly with leftward shift of the heart; recommend immediate surgical intervention.\nC) Depression of the lower half of the sternum with decreased space between the sternum and dorsal spine; consider calculating the pectus index to evaluate the need for surgery.\nD) Ill-defined right heart border simulating right middle lobe consolidation; suggest a follow-up CT scan for further evaluation.\nE) Prominence of the main pulmonary artery with loss of the descending aortic interface; initiate treatment for suspected pulmonary hypertension.\nF) Good delineation of the lower dorsal spine through the heart; conduct a cardiac MRI to assess for underlying congenital heart disease.' """

#image_path_list = ['/your/path/datasets/datasets--rajpurkarlab--ReXGradient-160K/snapshots/ce7001b4ccadba4b68bb0ede207eb8bc454c92f9/deid_png/GRDNBW2XZ2JAHL15/GRDNBXWFOSGOHN2U/studies/1.2.826.0.1.3680043.8.498.16648967710991011395607809297431316166/series/1.2.826.0.1.3680043.8.498.57797115066963102686166202562132901981/instances/1.2.826.0.1.3680043.8.498.10935343234754519619567585364610419609.png']
#question = 'Context: Age:72. Gender:F. Indication: Hypoxia, hypertension, diabetic, nonsmoker.\nQuestion: What is the most prominent cardiac finding visible on this chest X-ray?'

#image_path_list = ['/your/path/datasets/datasets--rajpurkarlab--ReXGradient-160K/snapshots/ce7001b4ccadba4b68bb0ede207eb8bc454c92f9/deid_png/GRDNO05X4EL65ZT5/GRDNFI8BE5J3S9AZ/studies/1.2.826.0.1.3680043.8.498.61246837883499103747735112982300111696/series/1.2.826.0.1.3680043.8.498.35740686676110646617531964362085026316/instances/1.2.826.0.1.3680043.8.498.39014434940073780814706353344536862883.png']
#question = 'Context: Age:30. Gender:M. Indication: Sudden onset of hypertension, chest pain..\nQuestion: What factor limits the detailed evaluation of the chest X-ray findings?'

""" image_path_list = ['/your/path/datasets/deid_png/GRDNDVCK4TE5ANF9/GRDNF262OG7L5ZFE/studies/1.2.826.0.1.3680043.8.498.88327049973162486901001479254471287181/series/1.2.826.0.1.3680043.8.498.31507761090396169563252714551327859765/instances/1.2.826.0.1.3680043.8.498.70883468531660859198819006202803518687.png']
question = 'Context: Age:Unknown. Gender:F. Indication: Emesis in a neonate. Feeding tube placement..\nQuestion: What is the most notable finding regarding lung volume on this chest X-ray?'

request_data = {
    "image_paths": image_path_list,
    "prompt": question
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

response = requests.post(
    "http://dgx-047:5007/medgemma",
    json=request_data,
    headers=headers,
    proxies=proxies,
    timeout=50
)
response.raise_for_status()

result = response.json()
print(result) """

#-----------------------------------------lingshu-------------------------------------------
""" {"extra_info": {"answer": "C", "index": 3, "split": "train"}}

image_path_list = ["figures/1801/figure_1a.jpg", "figures/1801/figure_1b.jpg"]
question = 'Based on the provided chest X-ray images, which of the following findings most strongly supports the diagnosis of pectus excavatum, and what is the next appropriate step in managing this condition?\n\nA) The presence of sloping anterior ends of the ribs; monitor the patient for symptoms of respiratory distress.\nB) Apparent cardiomegaly with leftward shift of the heart; recommend immediate surgical intervention.\nC) Depression of the lower half of the sternum with decreased space between the sternum and dorsal spine; consider calculating the pectus index to evaluate the need for surgery.\nD) Ill-defined right heart border simulating right middle lobe consolidation; suggest a follow-up CT scan for further evaluation.\nE) Prominence of the main pulmonary artery with loss of the descending aortic interface; initiate treatment for suspected pulmonary hypertension.\nF) Good delineation of the lower dorsal spine through the heart; conduct a cardiac MRI to assess for underlying congenital heart disease.' """

#image_path_list = ['/your/path/datasets/datasets--rajpurkarlab--ReXGradient-160K/snapshots/ce7001b4ccadba4b68bb0ede207eb8bc454c92f9/deid_png/GRDNBW2XZ2JAHL15/GRDNBXWFOSGOHN2U/studies/1.2.826.0.1.3680043.8.498.16648967710991011395607809297431316166/series/1.2.826.0.1.3680043.8.498.57797115066963102686166202562132901981/instances/1.2.826.0.1.3680043.8.498.10935343234754519619567585364610419609.png']
#question = 'Context: Age:72. Gender:F. Indication: Hypoxia, hypertension, diabetic, nonsmoker.\nQuestion: What is the most prominent cardiac finding visible on this chest X-ray?'

#image_path_list = ['/your/path/datasets/datasets--rajpurkarlab--ReXGradient-160K/snapshots/ce7001b4ccadba4b68bb0ede207eb8bc454c92f9/deid_png/GRDNO05X4EL65ZT5/GRDNFI8BE5J3S9AZ/studies/1.2.826.0.1.3680043.8.498.61246837883499103747735112982300111696/series/1.2.826.0.1.3680043.8.498.35740686676110646617531964362085026316/instances/1.2.826.0.1.3680043.8.498.39014434940073780814706353344536862883.png']
#question = 'Context: Age:30. Gender:M. Indication: Sudden onset of hypertension, chest pain..\nQuestion: What factor limits the detailed evaluation of the chest X-ray findings?'

#"gt_answer": "C"
image_path_list = ["/your/path/datasets/deid_png_8bit_new/GRDN7XC1B0NIVS9R/GRDNWPLFV2D04OWB/studies/1.2.826.0.1.3680043.8.498.46036170191488818044358209799322958552/series/1.2.826.0.1.3680043.8.498.57542431937587399428563927944584812859/instances/1.2.826.0.1.3680043.8.498.22092266386257130533250280196782513681.png", "/your/path/datasets/deid_png_8bit_new/GRDN7XC1B0NIVS9R/GRDNWPLFV2D04OWB/studies/1.2.826.0.1.3680043.8.498.46036170191488818044358209799322958552/series/1.2.826.0.1.3680043.8.498.96812740189087871660660065756628253013/instances/1.2.826.0.1.3680043.8.498.73776488740420197451885236176581536074.png"]
question = "Question:What is the status of lung volume on this chest X-ray?\nA) Increased lung volume\nB) Decreased lung volume\nC) Normal lung volume\nD) Hyperinflated lungs"

#"gt_answer": "A"
image_path_list = ["/your/path/datasets/deid_png_8bit_new/GRDN7AUMTSWQEEDN/GRDNSVLL301LNN64/studies/1.2.826.0.1.3680043.8.498.59011448955328377914389119538646170500/series/1.2.826.0.1.3680043.8.498.37639557907789340776446477996729278848/instances/1.2.826.0.1.3680043.8.498.50398178730996046106841098310123825322.png", "/your/path/datasets/deid_png_8bit_new/GRDN7AUMTSWQEEDN/GRDNSVLL301LNN64/studies/1.2.826.0.1.3680043.8.498.59011448955328377914389119538646170500/series/1.2.826.0.1.3680043.8.498.68671009238035880359219243716944042966/instances/1.2.826.0.1.3680043.8.498.49477317933690429805413643824649382990.png"]
question = "Question:Which of the following findings is absent on this chest X-ray?\nA) Focal infiltrate\nB) Central airway thickening\nC) Mild hyperinflation\nD) Pleural effusion"

request_data = {
    "image_path": image_path_list[0],
    "prompt": question
}
headers = {
    "Content-Type": "application/json"
}
proxies = {
    "http": None,
    "https": None
}

response = requests.post(
    "http://dgx-039:5006/lingshu",
    json=request_data,
    headers=headers,
    proxies=proxies,
    timeout=50
)
response.raise_for_status()

result = response.json()
print(result)