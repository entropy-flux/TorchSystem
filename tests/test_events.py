from pytest import raises
from unittest.mock import Mock
from torchsystem.domain import Event
from torchsystem.domain import Events

class ClsEvent(Event):...

class ObjEvent(Event):
    def __init__(self):
        self.value = 0

def test_events_with_exc():
    events = Events()
    events.enqueue(StopIteration)

    with raises(StopIteration):
        events.commit()

    mock_handler = Mock()
    events.handlers[StopIteration] = mock_handler
    events.enqueue(StopIteration)
    events.commit()
    mock_handler.assert_called_once()

def test_events():
    events = Events()
    events.enqueue(ClsEvent)

    mock_handler = Mock()
    events.handlers[ClsEvent] = mock_handler
    events.commit()

    mock_handler.assert_called_once()

def test_events_with_obj():
    events = Events()
    events.enqueue(ObjEvent())

    mock_handler = Mock()
    events.handlers[ObjEvent] = mock_handler
    events.commit()

    mock_handler.assert_called_once()
    called_args, called_kwargs = mock_handler.call_args
    assert isinstance(called_args[0], ObjEvent), "Expected an ObjEvent instance"

    events.enqueue(ClsEvent)
    events.commit() # No handler for ClsEvent DO NOTHING.

def test_enqueued():
    events = Events()
    events.enqueue(ClsEvent)
    events.enqueue(ClsEvent)
    events.enqueue(ClsEvent)
    events.enqueue(StopIteration)

    mock = Mock()
    events.handlers[ClsEvent] = mock

    assert len(events.queue) == 4
    with raises(StopIteration):
        events.commit()

    assert len(events.queue) == 0
    assert mock.call_count == 3

def test_multiple_handlers():
    events = Events()
    events.enqueue(ClsEvent)

    mock1 = Mock()
    mock2 = Mock()
    events.handlers[ClsEvent] = [mock1, mock2]
    events.commit()

    mock1.assert_called_once()
    mock2.assert_called_once()