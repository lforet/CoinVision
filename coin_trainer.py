import pygame


class Event(object):
    def __init__(self, name, kw={}):
        self.name = name
        self.__dict__.update(kw)
        self.dict = dict(kw)
        self.method_name = 'EVT_%s' % self.name
    
    def __repr__(self):
        return "<event>" % (self.name, self.dict)
 
 
class Dispatcher(object):
    def add(self, other):
        if isinstance(other, Dispatcher):
            other.parent = self
        else:
            raise ValueError("%s is not a Dispatcher" % other)
            
    def remove(self, other):
        if isinstance(other, Dispatcher):
            if other.parent is self:
                del other.parent
            else:
                raise ValueError("%s is not my child." % other)
        else:
            raise ValueError("%s is not a Dispatcher" % other)
        
    def dispatch(self, event):
        method = getattr(self, event.method_name, None)
        if method is not None: 
            return method(event)
        else:
            self.dispatch_to_parent(event)
    
    def dispatch_to_parent(self, event):
        parent = getattr(self, 'parent', None)
        if parent is not None:
            dispatch = getattr(parent, 'dispatch', None)
            if dispatch is not None: 
                return dispatch(event)
            
 
class Root(Dispatcher):
    def __init__(self):
        Dispatcher.__init__(self)
        self.focused_node = self
    
    def focus(self, node):
        if node is self.focused_node: return
        print 'focusing on', node
        self.focused_node.dispatch(Event('BLUR'))
        self.focused_node = node
        node.dispatch(Event('FOCUS'))
        
    def blur(self, node):
        self.focused_node = self
        node.dispatch(Event('BLUR'))
    
    def post(self, event_name, kw):
        event = Event(event_name, kw)
        return self.focused_node.dispatch(event)
 
 
if __name__ == "__main__":
    class Button(Dispatcher):
        def EVT_KeyDown(self, event):
            return self
    #create the root node, which is the reciever of all events.        
    r = Root()
    #create a button node
    b = Button()
    #add the button node to the application
    r.add(b)
    #give the button node focus, it will now receive events
    r.focus(b)
    #post a KeyDown event, which the button will recieve
    print r.post('KeyDown')
    #post a Quit event, which the button will not receive. 
    #this event will be passed onto the button parent
    print r.post('Quit')
    #the return value of the post method is the node which handled the event
        

