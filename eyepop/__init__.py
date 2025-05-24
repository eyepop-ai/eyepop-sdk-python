__version__ = '1.0.2'

from eyepop.eyepopsdk import EyePopSdk
from eyepop.worker.worker_jobs import WorkerJob as Job
from eyepop.visualize import EyePopPlot as Plot

__all__ = ['EyePopSdk', 'Job', 'Plot']
