try:
    from eyepop._version import version as __version__
except ImportError:
    __version__ = "0.0.0+unknown"

import eyepop.eyepopsdk
import eyepop.worker.worker_jobs
import eyepop.visualize

EyePopSdk = eyepopsdk.EyePopSdk
Job = eyepop.worker.worker_jobs.WorkerJob
Plot = visualize.EyePopPlot
