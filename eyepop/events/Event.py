# automatically generated by the FlatBuffers compiler, do not modify

# namespace: events

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Event(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsEvent(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Event()
        x.Init(buf, n + offset)
        return x

    # Event
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Event
    def Method(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Event
    def EventTimeEpochMs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint64Flags, o + self._tab.Pos)
        return 0

    # Event
    def XRequestId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Event
    def Result(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Event
    def Status(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    # Event
    def HostIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    # Event
    def PathIndex(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int16Flags, o + self._tab.Pos)
        return 0

    # Event
    def WaitMs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Event
    def ProcessMs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Event
    def BodyBytesSent(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Event
    def BodyBytesReceived(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def EventStart(builder): builder.StartObject(11)
def EventAddMethod(builder, method): builder.PrependInt8Slot(0, method, 0)
def EventAddEventTimeEpochMs(builder, eventTimeEpochMs): builder.PrependUint64Slot(1, eventTimeEpochMs, 0)
def EventAddXRequestId(builder, xRequestId): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(xRequestId), 0)
def EventAddResult(builder, result): builder.PrependInt8Slot(3, result, 0)
def EventAddStatus(builder, status): builder.PrependInt16Slot(4, status, 0)
def EventAddHostIndex(builder, hostIndex): builder.PrependInt16Slot(5, hostIndex, 0)
def EventAddPathIndex(builder, pathIndex): builder.PrependInt16Slot(6, pathIndex, 0)
def EventAddWaitMs(builder, waitMs): builder.PrependUint32Slot(7, waitMs, 0)
def EventAddProcessMs(builder, processMs): builder.PrependUint32Slot(8, processMs, 0)
def EventAddBodyBytesSent(builder, bodyBytesSent): builder.PrependUint32Slot(9, bodyBytesSent, 0)
def EventAddBodyBytesReceived(builder, bodyBytesReceived): builder.PrependUint32Slot(10, bodyBytesReceived, 0)
def EventEnd(builder): return builder.EndObject()
