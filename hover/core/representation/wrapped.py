"""
Wrapping dimensionality reduction classes to provide `fit_transform()` / `transform()` methods.
"""

class CVAEReducer:
    """
    ???+ note "Wrapping cvae.CompressionVAE."
    """
    def __init__(self, *args, **kwargs):
        """
        ???+ note "Keep the keyword arguments."
            
            Not expecting any args.
        """
        self.kwargs = kwargs.copy()
        
    def fit_transform(self, array):
        # lazy import for optional dependency
        from cvae import cvae
        
        self.embedder = cvae.CompressionVAE(X=array, **self.kwargs)
        self.embedder.train()
        
        embedding = self.embedder.embed(array)
        return embedding
    
    def transform(self, array):
        return self.embedder.embed(array)