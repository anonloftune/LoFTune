import inspect

import evaluate

from llms import LongMistralEval, LongGPTEval


class EvalMetricsFactory:
    @staticmethod
    def _load(cls, kwargs):
        cls_kwargs = {
            k for k, par in inspect.signature(cls.__init__).parameters.items()
        }
        valid_args = set(kwargs.keys()).intersection(cls_kwargs)
        valid_kwargs = {arg: kwargs[arg] for arg in valid_args}
        return cls(**valid_kwargs)

    @classmethod
    def load(cls, name_or_path, **kwargs):
        if name_or_path == "MistralEval":
            return cls._load(LongMistralEval, kwargs)
        elif name_or_path == "Factuality":
            return cls._load(Factuality, kwargs)
        elif name_or_path == "rougeL":
            return cls._load(RougeL, kwargs)
        elif name_or_path == "JsonStructureEvaluator":
            return cls._load(JsonStructureEvaluator, kwargs)
        elif name_or_path == "JsonExtractionEvaluator":
            return cls._load(JsonExtractionEvaluator, kwargs)
        elif name_or_path == "GPTEval":
            return cls._load(LongGPTEval, kwargs)
        else:
            try:
                return evaluate.load(name_or_path, **kwargs)
            except Exception as e:
                raise ValueError(f"Failed to load {name_or_path}, metric not found. Got exception {e}")
