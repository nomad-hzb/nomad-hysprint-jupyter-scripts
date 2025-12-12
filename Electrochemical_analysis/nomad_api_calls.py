import requests
import getpass


def init_cache():
    import requests_cache
    requests_cache.install_cache("my_local_cache", allowable_methods=('GET', 'POST'),
                                 ignored_parameters=['Authorization'])


def get_all_uploads(url, token, number_of_uploads=20):
    response = requests.get(f'{url}/uploads',
                            headers={'Authorization': f'Bearer {token}'},
                            params=dict(page_size=number_of_uploads, order_by='upload_create_time', order="desc"))
    return response.json()["data"]


def get_samples_in_upload(url, token, upload_id, entry_type="HySprint_Sample"):
    query = {
        'required': {
            'data': '*'
        },
        'owner': 'visible',
        'query': {'entry_type': entry_type,
                  'upload_id': upload_id},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/archive/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    data = response.json()["data"]
    return [d["archive"]["data"]["lab_id"] for d in data if "lab_id" in d["archive"]["data"]]


def get_template(url, token, upload_name, method):
    query = {
        'required': {
            'data': '*',
        },
        'owner': 'visible',
        'query': {"upload_name": upload_name, "entry_type": method},
        'pagination': {
            'page_size': 100
        }
    }
    response = requests.post(f'{url}/entries/archive/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    return response.json()["data"]


def get_token(url, name=None):
    user = name if name is not None else input("Username")
    print("Passwort: \n")
    password = getpass.getpass()
    
    #Get a token from the api, login
    response = requests.post(
        f'{url}/auth/token',
        data=dict(username=user, password=password, grant_type='password'),
    )
    return response.json()['access_token']


def get_batch_ids(url, token, batch_type="HySprint_Batch"):
    query = {
        'required': {
            'data': '*'
        },
        'owner': 'visible',
        'query': {'entry_type': batch_type},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/archive/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    data = response.json()["data"]
    return [d["archive"]["data"]["lab_id"] for d in data if "lab_id" in d["archive"]["data"]]


def get_ids_in_batch(url, token, batch_id, batch_type="HySprint_Batch"):
    query = {
        'required': {
            'data': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids': batch_id, 'entry_type': batch_type},
        'pagination': {
            'page_size': 100
        }
    }
    response = requests.post(
        f'{url}/entries/archive/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    data = response.json()["data"][0]
    sample_ids = []

    dd = data["archive"]["data"]
    if "entities" in dd:
        sample_ids.extend([s["lab_id"] for s in dd["entities"]])
    return sample_ids


def get_entry_data(url, token, entry_id):
    row = {"entry_id": entry_id}
    query = {
        'required': {
            'metadata': '*',
            'data': '*',
        },
        'owner': 'visible',
        'query': row,
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(f'{url}/entries/archive/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    assert len(response.json()["data"]) == 1, "Entry not found"
    return response.json()["data"][0]["archive"]["data"]


def get_sample_description(url, token, sample_ids):
    query = {
        'required': {
            'data': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids:any': sample_ids},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    entries = response.json()["data"]
    res = {}
    for entry in entries:
        data = entry["data"]
        res.update({data["lab_id"]: data.get("description", "")})
    return res


def get_entryid(url, token, sample_id):  # give it a batch id
    # get al entries related to this batch id
    query = {
        'required': {
            'metadata': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids': sample_id},
        'pagination': {
            'page_size': 100
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    data = response.json()["data"]
    assert len(data) == 1
    return data[0]["entry_id"]


def get_nomad_ids_of_entry(url, token, sample_id):  # give it a batch id
    # get al entries related to this batch id
    query = {
        'required': {
            'metadata': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids': sample_id},
        'pagination': {
            'page_size': 100
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    data = response.json()["data"]
    assert len(data) == 1
    return data[0]["entry_id"], data[0]["upload_id"]


def get_entry_meta_data(url, token, entry_id):
    row = {"entry_id": entry_id}
    query = {
        'required': {
            'metadata': '*',
        },
        'owner': 'visible',
        'query': row,
        'pagination': {
            'page_size': 100
        }
    }
    response = requests.post(f'{url}/entries/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    assert len(response.json()["data"]) == 1, "Entry not found"
    return response.json()["data"][0]


def get_information(url, token, entry_id, path):
    mdata = get_entry_meta_data(url, token, entry_id)
    res = []
    for ref in mdata.get("entry_references"):
        if path != ref.get("source_path"):
            continue
        res.append(get_entry_data(url, token, ref.get("target_entry_id")))

    return res


def get_setup(url, token, entry_id):
    data = get_information(url, token, entry_id, "data.setup")
    assert data and len(data) == 1, "No Setup found"
    return data[0]


def get_environment(url, token, entry_id):
    data = get_information(url, token, entry_id, "data.environment")
    assert data and len(data) == 1, "No Environment found"
    return data[0]


def get_samples(url, token, entry_id):
    data = get_information(url, token, entry_id, "data.samples.reference")
    assert data and len(data) > 0, "No Samples found"
    return data


def get_specific_data_of_sample(url, token, sample_id, entry_type, with_meta=False):
    # collect the results of the sample, in this case it are all the annealing temperatures
    entry_id = get_entryid(url, token, sample_id)

    query = {
        'required': {
            'metadata': '*',
            'data': '*',
        },
        'owner': 'visible',
        'query': {'entry_references.target_entry_id': entry_id, 'entry_type': entry_type},
        'pagination': {
            'page_size': 100
        }
    }
    response = requests.post(f'{url}/entries/archive/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    linked_data = response.json()["data"]
    res = []
    for ldata in linked_data:
        if with_meta:
            res.append((ldata["archive"]["data"], ldata["archive"]["metadata"]))
        else:
            res.append(ldata["archive"]["data"])
    return res


def get_all_JV(url, token, sample_ids, jv_type="HySprint_JVmeasurement"):
    # collect the results of the sample, in this case it are all the annealing temperatures
    query = {
        'required': {
            'metadata': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids:any': sample_ids},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)

    entry_ids = [entry["entry_id"] for entry in response.json()["data"]]

    query = {
        'required': {
            'data': '*',
            'metadata': '*',
        },
        'owner': 'visible',
        'query': {'entry_references.target_entry_id:any': entry_ids,
                  'entry_type': jv_type},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(f'{url}/entries/archive/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    linked_data = response.json()["data"]
    res = {}
    for ldata in linked_data:
        lab_id = ldata["archive"]["data"]["samples"][0]["lab_id"]
        if lab_id not in res:
            res[lab_id] = []
        res[lab_id].append((ldata["archive"]["data"], ldata["archive"]["metadata"]))
    return res


def get_all_measurements_except_JV(url, token, sample_ids):
    # collect the results of the sample, in this case it are all the annealing temperatures
    query = {
        'required': {
            'metadata': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids:any': sample_ids},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)

    entry_ids = [entry["entry_id"] for entry in response.json()["data"]]

    query = {
        'required': {
            'data': '*',
            'metadata': '*',
        },
        'owner': 'visible',
        'query': {'entry_references.target_entry_id:any': entry_ids,
                  'section_defs.definition_qualified_name': 'baseclasses.BaseMeasurement'},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(f'{url}/entries/archive/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    linked_data = response.json()["data"]
    res = {}
    for ldata in linked_data:
        if "entry_type" not in ldata["archive"]["metadata"] or "JV" in ldata["archive"]["metadata"]["entry_type"]:
            continue
        lab_id = ldata["archive"]["data"]["samples"][0]["lab_id"]
        if lab_id not in res:
            res[lab_id] = []
        res[lab_id].append((ldata["archive"]["data"], ldata["archive"]["metadata"]))
    return res


def get_all_measurements_except_JV(url, token, sample_ids):
    # collect the results of the sample, in this case it are all the annealing temperatures
    query = {
        'required': {
            'metadata': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids:any': sample_ids},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)

    entry_ids = [entry["entry_id"] for entry in response.json()["data"]]

    query = {
        'required': {
            'data': '*',
            'metadata': '*',
        },
        'owner': 'visible',
        'query': {'entry_references.target_entry_id:any': entry_ids,
                  'section_defs.definition_qualified_name': 'baseclasses.BaseMeasurement'},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(f'{url}/entries/archive/query',
                             headers={'Authorization': f'Bearer {token}'}, json=query)
    linked_data = response.json()["data"]
    res = {}
    for ldata in linked_data:
        if "entry_type" not in ldata["archive"]["metadata"] or "JV" in ldata["archive"]["metadata"]["entry_type"]:
            continue
        lab_id = ldata["archive"]["data"]["samples"][0]["lab_id"]
        if lab_id not in res:
            res[lab_id] = []
        res[lab_id].append((ldata["archive"]["data"], ldata["archive"]["metadata"]))
    return res

def download_file(url,upload_id,file_name,token):
    # file_name = file_name.replace("#","%23")
    response = requests.get(
        f'{url}/uploads/{upload_id}/raw/{file_name}', headers={'Authorization': f'Bearer {token}'},json=dict(ignore_mime_type=True))
    return response.content


def get_sample_description(url, token, sample_ids):
    query = {
        'required': {
            'data': '*'
        },
        'owner': 'visible',
        'query': {'results.eln.lab_ids:any': sample_ids},
        'pagination': {
            'page_size': 10000
        }
    }
    response = requests.post(
        f'{url}/entries/query', headers={'Authorization': f'Bearer {token}'}, json=query)
    entries = response.json()["data"]
    res = {}
    for entry in entries:
        data = entry["data"]
        res.update({data["lab_id"]: data.get("description", "")})
    return res