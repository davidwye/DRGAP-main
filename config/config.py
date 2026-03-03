from utils.file_utils import yaml_load, json_readb

config_data = yaml_load('config/config.yaml')
root = config_data['root']
modelMap = config_data['modelConfig']
datasetsPath = config_data['datasetsConfig']
DRGAPPromptTemplates = json_readb(config_data['promptConfig']['DRGAPTemplatePath'])
QueryPromptTemplates = json_readb(config_data['promptConfig']['QueryTemplatePath'])
