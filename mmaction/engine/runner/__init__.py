# Copyright (c) OpenMMLab. All rights reserved.
from .multi_loop import MultiLoaderEpochBasedTrainLoop
from .retrieval_loop import RetrievalTestLoop, RetrievalValLoop
from .timed_eval_loop import TimedTestLoop, TimedValLoop

__all__ = [
    'MultiLoaderEpochBasedTrainLoop', 'RetrievalValLoop', 'RetrievalTestLoop',
    'TimedValLoop', 'TimedTestLoop'
]
