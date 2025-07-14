#
# Copyright 2015 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from collections.abc import Iterable
from copy import copy
import warnings
from datetime import tzinfo, time
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING
import logbook
import pytz
import pandas as pd
import numpy as np

from itertools import chain, repeat

from six import (
    exec_,
    iteritems,
    itervalues,
    string_types,
)
from trading_calendars.utils.pandas_utils import days_at_time
from trading_calendars import get_calendar

from zipline._protocol import handle_non_market_minutes
from zipline.errors import (
    AttachPipelineAfterInitialize,
    CannotOrderDelistedAsset,
    DuplicatePipelineName,
    HistoryInInitialize,
    IncompatibleCommissionModel,
    IncompatibleSlippageModel,
    NoSuchPipeline,
    OrderDuringInitialize,
    OrderInBeforeTradingStart,
    PipelineOutputDuringInitialize,
    RegisterAccountControlPostInit,
    RegisterTradingControlPostInit,
    ScheduleFunctionInvalidCalendar,
    SetBenchmarkOutsideInitialize,
    SetCancelPolicyPostInit,
    SetCommissionPostInit,
    SetSlippagePostInit,
    UnsupportedCancelPolicy,
    UnsupportedDatetimeFormat,
    UnsupportedOrderParameters,
    ZeroCapitalError
)
from zipline.finance.blotter import SimulationBlotter
from zipline.finance.controls import (
    LongOnly,
    MaxOrderCount,
    MaxOrderSize,
    MaxPositionSize,
    MaxLeverage,
    MinLeverage,
    RestrictedListOrder
)
from zipline.finance.execution import (
    LimitOrder,
    MarketOrder,
    StopLimitOrder,
    StopOrder,
)
from zipline.finance.asset_restrictions import Restrictions
from zipline.finance.cancel_policy import NeverCancel, CancelPolicy
from zipline.finance.asset_restrictions import (
    NoRestrictions,
    StaticRestrictions,
    SecurityListRestrictions,
)
from zipline.assets import Asset, Equity, Future
from zipline.gens.tradesimulation import AlgorithmSimulator
from zipline.finance.metrics import MetricsTracker, load as load_metrics_set
from zipline.pipeline import Pipeline
import zipline.pipeline.domain as domain
from zipline.pipeline.engine import (
    ExplodingPipelineEngine,
    SimplePipelineEngine,
)
from zipline.utils.api_support import (
    api_method,
    require_initialized,
    require_not_initialized,
    ZiplineAPI,
    disallowed_in_before_trading_start)
from zipline.utils.compat import ExitStack
from zipline.utils.input_validation import (
    coerce_string,
    ensure_upper_case,
    error_keywords,
    expect_dtypes,
    expect_types,
    optional,
    optionally,
)
from zipline.utils.numpy_utils import int64_dtype
from zipline.utils.pandas_utils import normalize_date
from zipline.utils.cache import ExpiringCache
from zipline.utils.pandas_utils import clear_dataframe_indexer_caches

import zipline.utils.events
from zipline.utils.events import (
    EventManager,
    make_eventrule,
    date_rules,
    time_rules,
    calendars,
    AfterOpen,
    BeforeClose
)
from zipline.utils.math_utils import (
    tolerant_equals,
    round_if_near_integer,
)
from zipline.utils.preprocess import preprocess
from zipline.utils.security_list import SecurityList

import zipline.protocol
from zipline.sources.requests_csv import PandasRequestsCSV

from zipline.gens.sim_engine import MinuteSimulationClock
from zipline.sources.benchmark_source import BenchmarkSource
from zipline.zipline_warnings import ZiplineDeprecationWarning

if TYPE_CHECKING:
    from zipline.data.data_portal import DataPortal
    from zipline.assets.assets import AssetFinder
    from zipline.finance.metrics import MetricsSet
    from zipline.utils.calendar import TradingCalendar

log = logbook.Logger("ZiplineLog")

# For creating and storing pipeline instances
AttachedPipeline = namedtuple('AttachedPipeline', 'pipe chunks eager')


class NoBenchmark(ValueError):
    def __init__(self):
        super(NoBenchmark, self).__init__(
            'Must specify either benchmark_sid or benchmark_returns.',
        )


class TradingAlgorithm:
    """A class that represents a trading strategy and parameters to execute
    the strategy.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to ``initialize`` unless listed below.
    initialize : callable[context -> None], optional
        Function that is called at the start of the simulation to
        setup the initial context.
    handle_data : callable[(context, data) -> None], optional
        Function called on every bar. This is where most logic should be
        implemented.
    before_trading_start : callable[(context, data) -> None], optional
        Function that is called before any bars have been processed each
        day.
    analyze : callable[(context, DataFrame) -> None], optional
        Function that is called at the end of the backtest. This is passed
        the context and the performance results for the backtest.
    script : str, optional
        Algoscript that contains the definitions for the four algorithm
        lifecycle functions and any supporting code.
    namespace : dict, optional
        The namespace to execute the algoscript in. By default this is an
        empty namespace that will include only python built ins.
    algo_filename : str, optional
        The filename for the algoscript. This will be used in exception
        tracebacks. default: '<string>'.
    data_frequency : {'daily', 'minute'}, optional
        The duration of the bars.
    equities_metadata : dict or DataFrame or file-like object, optional
        If dict is provided, it must have the following structure:
        * keys are the identifiers
        * values are dicts containing the metadata, with the metadata
          field name as the key
        If pandas.DataFrame is provided, it must have the
        following structure:
        * column names must be the metadata fields
        * index must be the different asset identifiers
        * array contents should be the metadata value
        If an object with a ``read`` method is provided, ``read`` must
        return rows containing at least one of 'sid' or 'symbol' along
        with the other metadata fields.
    futures_metadata : dict or DataFrame or file-like object, optional
        The same layout as ``equities_metadata`` except that it is used
        for futures information.
    identifiers : list, optional
        Any asset identifiers that are not provided in the
        equities_metadata, but will be traded by this TradingAlgorithm.
    get_pipeline_loader : callable[BoundColumn -> PipelineLoader], optional
        Function that takes a BoundColumn and returns a PipelineLoader
        to use for that column. If not provided, will use the default
        loader for the column's dataset.
    """

    def __init__(self,
                 sim_params,
                 data_portal: Optional[DataPortal] = None,
                 asset_finder: Optional[AssetFinder] = None,
                 # Algorithm API
                 namespace: Optional[Dict[str, Any]] = None,
                 script: Optional[str] = None,
                 algo_filename: Optional[str] = None,
                 initialize: Optional[Callable] = None,
                 handle_data: Optional[Callable] = None,
                 before_trading_start: Optional[Callable] = None,
                 analyze: Optional[Callable] = None,
                 #
                 trading_calendar: Optional[TradingCalendar] = None,
                 metrics_set: Optional[MetricsSet] = None,
                 blotter: Optional[SimulationBlotter] = None,
                 blotter_class: Optional[type] = None,
                 cancel_policy: Optional[CancelPolicy] = None,
                 benchmark_sid: Optional[Asset] = None,
                 benchmark_returns: Optional[pd.Series] = None,
                 platform: str = 'zipline',
                 capital_changes: Optional[pd.Series] = None,
                 get_pipeline_loader: Optional[Callable] = None,
                 create_event_context: Optional[Callable] = None,
                 **initialize_kwargs):
        # List of trading controls to be used to validate orders.
        self.trading_controls = []

        # List of account controls to be checked on each bar.
        self.account_controls = []

        # List of restrictions to be checked on each order.
        self.restrictions = NoRestrictions()

        # The simulation parameters for this algorithm.
        self.sim_params = sim_params

        # The data portal that will serve data to the algorithm.
        self.data_portal = data_portal

        # The asset finder that will resolve assets for the algorithm.
            self.asset_finder = asset_finder

        # The trading calendar for the algorithm.
        self.trading_calendar = trading_calendar

        # The metrics set that will track performance.
        self.metrics_set = metrics_set

        # The blotter that will track orders and positions.
            self.blotter = blotter

        # The blotter class to use if no blotter is provided.
        self.blotter_class = blotter_class

        # The cancel policy for the algorithm.
        self.cancel_policy = cancel_policy

        # The benchmark asset or returns series.
        self.benchmark_sid = benchmark_sid
        self.benchmark_returns = benchmark_returns

        # The platform identifier.
        self.platform = platform

        # Capital changes over time.
        self.capital_changes = capital_changes

        # Function to get pipeline loaders.
        self.get_pipeline_loader = get_pipeline_loader

        # Function to create event context.
        self.create_event_context = create_event_context

        # The namespace for the algorithm.
        self.namespace = namespace or {}

        # The script for the algorithm.
        self.script = script

        # The filename for the algorithm.
        self.algo_filename = algo_filename

        # The algorithm lifecycle functions.
        self.initialize_func = initialize
        self.handle_data_func = handle_data
        self.before_trading_start_func = before_trading_start
        self.analyze_func = analyze

        # The initialize keyword arguments.
        self.initialize_kwargs = initialize_kwargs

        # Whether the algorithm has been initialized.
        self.initialized = False

        # The event manager for the algorithm.
        self.event_manager = EventManager()

        # The pipeline engine for the algorithm.
        self.pipeline_engine = None

        # The attached pipelines.
        self.attached_pipelines = {}

        # The recorded variables.
        self.recorded_vars = {}

        # The logger for the algorithm.
        self.logger = logbook.Logger('Algorithm')

        # The current datetime.
        self.current_dt = None

        # The data frequency.
        self.data_frequency = None

        # The benchmark source.
        self.benchmark_source = None

        # The metrics tracker.
            self.metrics_tracker = None

        # The generator for the algorithm.
        self.generator = None

        # The universe function.
        self._universe_func = None

        # The last calculated universe.
        self._last_calculated_universe = None

        # The universe last updated at.
        self._universe_last_updated_at = None

        # Initialize the algorithm if functions are provided.
        if any([initialize, handle_data, before_trading_start, analyze]):
            self.initialize(*args, **kwargs)

    def initialize(self, *args, **kwargs):
        """Initialize the algorithm with the provided arguments."""
        if self.initialized:
            raise RuntimeError("Algorithm already initialized")
        
        if self.initialize_func:
            self.initialize_func(self, *args, **kwargs)
        
        self.initialized = True

    def before_trading_start(self, data):
        """Called before trading starts each day."""
        if self.before_trading_start_func:
            self.before_trading_start_func(self, data)

    def handle_data(self, data):
        """Called on each bar with the current data."""
        if self.handle_data_func:
            self.handle_data_func(self, data)

    def analyze(self, perf):
        """Called at the end of the backtest with performance results."""
        if self.analyze_func:
            self.analyze_func(self, perf)

    def __repr__(self):
        return f"TradingAlgorithm(initialized={self.initialized})"

    # ... rest of the implementation continues with modernized methods
