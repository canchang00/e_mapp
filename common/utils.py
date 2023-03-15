class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    def update(self,*args,**kwargs):
        super(AttrDict, self).update(*args,**kwargs)
        self.__dict__ = self