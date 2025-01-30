from torchsystem.domain import Events, Event

class ClsEvent(Event):...

class ObjEvent(Event):
    def __init__(self, value):
        self.value = value

class OtherObjEvent(Event):
    def __init__(self, willbeignored):
        self.value = willbeignored

events = Events()
events.enqueue(ClsEvent)
events.enqueue(KeyError) # Enqueues a KeyError exception event
events.enqueue(ObjEvent('somevalue'))
events.enqueue(OtherObjEvent('willbeignored'))
events.enqueue(StopIteration) # Enqueues a StopIteration exception event

events.handlers[ClsEvent] = lambda: print('ClsEvent was handled.')
events.handlers[KeyError] = lambda: print('KeyError was handled.')
events.handlers[ObjEvent] = lambda event: print(f'ObjEvent was handled with value: {event.value}')
events.handlers[OtherObjEvent] = lambda: print('OtherObjEvent was handled.')

try:
    events.commit()
except StopIteration:
    print('StopIteration exception was raised.')

# Output:
#ClsEvent was handled.
#KeyError was handled.
#ObjEvent was handled with value: somevalue
#OtherObjEvent was handled.
#StopIteration exception was raised. Usefull for early stopping in training loops.