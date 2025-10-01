class SafeLLM:
    def __init__(self, llm, reasoning_model=True):
        self.llm = llm
        self.reasoning_model = reasoning_model

    def bind(self, **kwargs):
        # Ajuste específico para evitar problemas con `stop`
        if self.reasoning_model and "stop" in kwargs:
            kwargs.pop("stop")
        return self.llm.bind(**kwargs)

    def __getattr__(self, name):
        """
        Si el método no está definido en SafeLLM,
        lo delegamos al LLM interno.
        """
        return getattr(self.llm, name)
