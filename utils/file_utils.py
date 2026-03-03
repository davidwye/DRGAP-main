import json
import os
import yaml
def tsv_read(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()
        data = []
        for line in lines:
            fields = line.strip().split('\t')
            data.append(fields)
        return data[0], data[1:]

def csv_read(filepath):
    import csv
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            data.append(row)
        return data[0], data[1:]

def json_readb(filename):
    with open(filename, "rb") as f:
        data = json.load(f)
    return data

def json_read(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def jsonl_read(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data_dict = json.loads(line)
            data.append(data_dict)
    return data

def jsonl_write(filename,data):
    try:
        with open(filename, 'a') as file:
            json.dump(data, file)
            file.write("\n")
    except Exception as e:
        print("jsonl write error: ", e)


def find_files_with_string(directory, search_string):
    found_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if search_string in file:
                found_files.append(os.path.join(root, file))

    return found_files

def yaml_load(filename):

    with open(filename, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    return data

