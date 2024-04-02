import yaml
def read_config_file():
    """
    Read the config.yaml file

    """
    with open("config/config.yaml") as file:
        config_data = yaml.safe_load(file)
    return config_data