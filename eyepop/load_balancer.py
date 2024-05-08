import random
import time
from threading import Lock


class EndpointEntry:
    def __init__(self, endpoint, mutex):
        self.mutex = mutex
        self.base_url = endpoint['base_url']
        self.pipeline_id = endpoint['pipeline_id']
        self.last_error_time = None

    def mark_error(self):
        with self.mutex:
            self.last_error_time = time.time()

    def mark_success(self):
        with self.mutex:
            self.last_error_time = None


class EndpointLoadBalancer:
    def __init__(self, endpoints):
        self.mutex = Lock()
        self.entries = []
        for endpoint in endpoints:
            entry = EndpointEntry(endpoint, self.mutex)
            self.entries.append(entry)
        if len(self.entries) > 0:
            self.next_index = random.randint(0, len(self.entries)-1)
        else:
            self.next_index = -1

    def get_debug_status(self) -> list[dict]:
        statuss = []
        for entry in self.entries:
            status = {
                'base_url': entry.base_url,
                'pipeline_id': entry.pipeline_id,
                'last_error': entry.last_error_time
            }
            statuss.append(status)
        return statuss

    def next_entry(self, retry_after_secs: float) -> EndpointEntry | None:
        with self.mutex:
            now = time.time()
            i = 0
            entry = None
            while i < len(self.entries):
                entry = self.entries[(self.next_index + i) % len(self.entries)]
                if entry.last_error_time is None or entry.last_error_time < now - retry_after_secs:
                    self.next_index = (self.next_index + i + 1) % len(self.entries)
                    break
                else:
                    entry = None
                    i += 1
            return entry
