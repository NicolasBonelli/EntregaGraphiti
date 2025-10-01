from src.config.config_azure import GraphitiConnector

_connector = None

def get_connector():
    global _connector
    if _connector is None:
        _connector = GraphitiConnector()
    return _connector