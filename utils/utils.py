import hashlib

def calculate_group_hash(diseases: list) -> str:
    """
    Generate hash value for everyone disease
    """
    if len(diseases):
        sorted_diseases = sorted(diseases)
        disease_tuple = tuple(sorted_diseases)
        return hashlib.md5(str(disease_tuple).encode()).hexdigest()
    else:
        return None
    
class GroupCache:
    cache = {}

    @classmethod
    def clear(cls):
        cls.cache.clear()