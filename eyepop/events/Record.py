# automatically generated by the FlatBuffers compiler, do not modify

# namespace: events

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Record(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsRecord(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Record()
        x.Init(buf, n + offset)
        return x

    # Record
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Record
    def ClientType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # Record
    def ClientVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Record
    def Hosts(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Record
    def HostsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Record
    def HostsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # Record
    def Paths(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Record
    def PathsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Record
    def PathsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        return o == 0

    # Record
    def Events(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from eyepop.events.Event import Event
            obj = Event()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Record
    def EventsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Record
    def EventsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        return o == 0

def RecordStart(builder): builder.StartObject(5)
def RecordAddClientType(builder, clientType): builder.PrependInt8Slot(0, clientType, 0)
def RecordAddClientVersion(builder, clientVersion): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(clientVersion), 0)
def RecordAddHosts(builder, hosts): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(hosts), 0)
def RecordStartHostsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def RecordAddPaths(builder, paths): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(paths), 0)
def RecordStartPathsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def RecordAddEvents(builder, events): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(events), 0)
def RecordStartEventsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def RecordEnd(builder): return builder.EndObject()