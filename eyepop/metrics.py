import logging
import time
from enum import Enum

from eyepop.jobs import JobState
from eyepop.worker.worker_jobs import JobStateCallback, WorkerJob

log = logging.getLogger('eyepop')

class MetricCollector(JobStateCallback):
    def __init__(self):
        self.jobs_to_state = {}
        self.jobs_to_last_updated = {}
        self.total_number_of_jobs = 0
        self.max_number_of_jobs_by_state = {JobState.STARTED: 0, JobState.IN_PROGRESS: 0, JobState.FINISHED: 0,
                                            JobState.DRAINED: 0, JobState.FAILED: 0, }
        self.number_of_jobs_reached_state = {JobState.IN_PROGRESS: 0, JobState.FINISHED: 0, JobState.DRAINED: 0,
                                             JobState.FAILED: 0, }
        self.total_time_to_reached_state = {JobState.IN_PROGRESS: 0.0, JobState.FINISHED: 0.0, JobState.DRAINED: 0.0,
                                            JobState.FAILED: 0.0, }

    def get_average_time(self, state: JobState) -> float:
        if self.number_of_jobs_reached_state[state] == 0:
            return 0.0
        else:
            return self.total_time_to_reached_state[state] / self.number_of_jobs_reached_state[state]

    def get_average_times(self) -> dict:
        times = {JobState.IN_PROGRESS: self.get_average_time(JobState.IN_PROGRESS),
                 JobState.FINISHED: self.get_average_time(JobState.FINISHED),
                 JobState.DRAINED: self.get_average_time(JobState.DRAINED),
                 JobState.FAILED: self.get_average_time(JobState.FAILED), }
        return times

    def update_count_by_state(self, state: JobState):
        count = 0
        for job, job_state in self.jobs_to_state.items():
            if state == job_state:
                count += 1
        if count > self.max_number_of_jobs_by_state[state]:
            self.max_number_of_jobs_by_state[state] = count

    def collect_execution_time(self, job: WorkerJob, new_state: JobState):
        now = time.time()
        duration = now - self.jobs_to_last_updated[job]
        self.jobs_to_last_updated[job] = now
        self.number_of_jobs_reached_state[new_state] += 1
        self.total_time_to_reached_state[new_state] += duration

    def created(self, job):
        self.total_number_of_jobs += 1
        self.jobs_to_state[job] = JobState.CREATED
        self.jobs_to_last_updated[job] = time.time()

    def started(self, job):
        current_state = self.jobs_to_state[job]
        if current_state != JobState.CREATED:
            log.debug("invalid job state %v in metrics collector for started(%v)", current_state, job)
        else:
            self.jobs_to_state[job] = JobState.STARTED
        self.jobs_to_last_updated[job] = time.time()
        self.update_count_by_state(JobState.STARTED)

    def first_result(self, job):
        current_state = self.jobs_to_state[job]
        if current_state != JobState.STARTED:
            log.debug("invalid job state %v in metrics collector for first_result(%v)", current_state, job)
        else:
            self.jobs_to_state[job] = JobState.IN_PROGRESS
        self.collect_execution_time(job, JobState.IN_PROGRESS)
        self.update_count_by_state(JobState.IN_PROGRESS)

    def failed(self, job):
        self.jobs_to_state[job] = JobState.FAILED
        self.collect_execution_time(job, JobState.FAILED)
        self.update_count_by_state(JobState.FAILED)
        self.jobs_to_last_updated[job] = time.time()

    def finished(self, job):
        self.jobs_to_state[job] = JobState.FINISHED
        self.collect_execution_time(job, JobState.FINISHED)
        self.update_count_by_state(JobState.FINISHED)
        self.jobs_to_last_updated[job] = time.time()

    def drained(self, job):
        self.jobs_to_state[job] = JobState.DRAINED
        self.collect_execution_time(job, JobState.DRAINED)
        self.update_count_by_state(JobState.DRAINED)
        self.jobs_to_last_updated[job] = time.time()

    def finalized(self, job):
        del self.jobs_to_state[job]


class JobState(Enum):
    CREATED = 1
    STARTED = 2
    IN_PROGRESS = 3
    FINISHED = 4
    FAILED = 5
    DRAINED = 6

    def __repr__(self):
        return self._name_
