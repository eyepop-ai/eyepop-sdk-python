try:
    from eyepop._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

import eyepop.logging
from eyepop import eyepopsdk
from eyepop.worker import worker_jobs
from eyepop import visualize

EyePopSdk = eyepopsdk.EyePopSdk
Job = worker_jobs.WorkerJob
Plot = visualize.EyePopPlot
