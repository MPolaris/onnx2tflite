class Registry(object):
    def __init__(self, name) -> None:
        self._name = name
        self._operator_dict = dict()

    def __len__(self):
        return len(self._operator_dict)

    @property
    def name(self):
        return self._name

    @property
    def operator_dict(self):
        return self._operator_dict

    def get(self, key):
        return self._operator_dict.get(key, None)

    def _register_operator(self, op_class, op_name=None):
        if (not isinstance(op_name, str)) or op_name is None:
            op_name = op_class.__name__
        
        if self._operator_dict.get(op_name, None):
            raise KeyError(f'{op_name} is already registered in {self._name}')
        
        self._operator_dict[op_name] = op_class

    def register_operator(self, name=None, op_class=None):
        if op_class is not None:
            self._register_operator(op_class, name)
            return op_class

        def _register(cls):
            self._register_operator(cls, name)
            return cls

        return _register

OPERATOR = Registry("TensorflowOP")