from typing import Union, List
import astropy.units as u
from sunpy.net import Fido
from sunpy.net import attrs
from datetime import datetime

from sunpy.net.vso import VSOQueryResponseTable
from data_collection.vso_search_result import VSOSearchResult
from utils.vso_instrument import VSOInstrument
from utils.search_utils import find_equally_spaced_indices, \
    partition_time_interval
from utils.search_utils import validate_time_interval

NSO_CAP_QUERY_SIZE = 10000


def sample_by_cadence(start_time: Union[datetime, str],
                      end_time: Union[datetime, str],
                      cadence: int,
                      instrument: Union[VSOInstrument, str],
                      verbose: bool = False) -> [VSOSearchResult]:
    """

    Parameters
    ----------
    start_time
        the start time of the search query (format: "%Y-%m-%d %H:%M:%S")
    end_time
        the end time of the search query (format: "%Y-%m-%d %H:%M:%S")
    cadence
        sampling cadence in minutes
    instrument
        the desired instrument (see 'vso_instrument.VSOInstrument' for the
        possible names.)
    verbose

    Returns
    -------
        a list of `VSOSearchResult` objects each for a different sub-intervals
        of the given query time
    """
    s_time, e_time = validate_time_interval(start_time, end_time)
    time_q = attrs.Time(s_time, e_time)
    inst_q = VSOInstrument(instrument).inst_obj

    results = Fido.search(time_q, inst_q,
                          attrs.Physobs.intensity,
                          attrs.Sample(cadence * u.minute))
    # --------------------------------------
    # if the query response is within the cap size
    # --------------------------------------
    if len(results[0]) < NSO_CAP_QUERY_SIZE:
        # the query has not reached the cap size
        res: VSOQueryResponseTable = results[0]
        return [VSOSearchResult(res)]

    # --------------------------------------
    # if the query response has reached the cap size
    # --------------------------------------
    if verbose:
        print(
            """
            The query response is inaccurate due to the hard-coded upper 
            limit (10,000) for the number of observations that can be queried
            in one attempt. This method will continue by breaking the given 
            interval into smaller sub-intervals and sending queries one at a 
            time. Of course, this results in a longer waiting time."""
        )
    all_vsos = []
    exp_population_size = int((e_time - s_time).total_seconds() // 60)
    exp_sample_size = exp_population_size // cadence
    # expected sample size will certainly be greater than 10k because this is
    # always north of the actual sampler size which already exceeded the cap
    n_intervals = exp_sample_size // NSO_CAP_QUERY_SIZE

    # form non-overlapping batches
    intervals = partition_time_interval(start_time=s_time,
                                        end_time=e_time,
                                        n_partitions=n_intervals)
    if verbose:
        print(
            """The queried interval was broken into {} sub-intervals."""
            .format(n_intervals))

    # search each batch with 1-min cadence
    for i in range(n_intervals):
        s_time, e_time = validate_time_interval(intervals[i], intervals[i+1])
        time_q = attrs.Time(s_time, e_time)
        if verbose:
            print(
                """\t> query interval: {} -- {}""".format(s_time, e_time)
            )
        results = Fido.search(time_q, inst_q,
                              attrs.Physobs.intensity,
                              attrs.Sample(cadence * u.minute))
        res: VSOQueryResponseTable = results[0]
        if verbose:
            print('\t\t\t> {} instances queried.'.format(len(res)))
        all_vsos.append(VSOSearchResult(res))
    return all_vsos
