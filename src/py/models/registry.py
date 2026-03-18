# models/registry.py

MODELS = {}

def register_model(name):
    """
    Dekorator do rejestrowania modeli w głównym słowniku.
    """
    def decorator(cls):
        MODELS[name] = cls
        return cls
    return decorator


def build_model(config):
    """
    Inicjalizuje i zwraca model na podstawie pliku konfiguracyjnego YAML.
    """
    model_name = config['model']['name']
    
    # Pobieramy argumenty modelu z sekcji kwargs w YAML (jeśli istnieją)
    model_kwargs = config['model'].get('kwargs', {})
    
    if model_name not in MODELS:
        raise ValueError(
            f"Model '{model_name}' nie jest zarejestrowany! "
            f"Dostępne modele: {list(MODELS.keys())}"
        )
        
    # Inicjalizacja modelu rozpakowanymi parametrami (**kwargs)
    return MODELS[model_name](**model_kwargs)
