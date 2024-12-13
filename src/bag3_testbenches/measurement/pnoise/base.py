# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Any, Optional, Mapping, List, Union, Tuple, Sequence

from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict

from ..pss.base import PSSTB


class PNoiseTB(PSSTB):
    """This class provide utility methods useful for all PNoise simulations.

    Use `sweep_options` to set the noise frequency sweep.
    Use `probe_info_list` to set one or more pnoise runs with the respective terminals and options.
    For sampled jitter, include the keyword `measurements` in `pnoise_options`. This must contain a
        list of dictionaries that will be parsed as `JitterEvent` objects. See `bag.simulation.data`
        for more details.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sweep_options : Mapping[str, Any]
        Dictionary with following entries :
            type : str
                type of sweep (LINEAR / LOG)
            start : Union[str, float]
                initial value of sweep_var
            stop : Union[str, float]
                final value of sweep_var
            num : int
                number of sweep data points
    probe_info_list : List[Tuple]
        A list of pnoise measurement info (p_port, n_port, pnoise options).
    pnoise_options : Optional[Mapping[str, Any]]
        Optional.  PNoise simulation options dictionary. Run `spectre -h pnoise` to see available parameters.
        If the key `measurement` is included, the contents will be parsed as `JitterEvent`s. See 
            `bag.simulation.data` for details.
    """

    def get_netlist_info(self) -> SimNetlistInfo:
        sim_setup = self.get_pss_sim_setup()
        sweep_options: Mapping[str, Any] = self.specs['sweep_options']
        probe_info_list: List[Tuple] = self.specs['probe_info_list']
        if len(probe_info_list) == 0:
            raise ValueError("At least one probe must be specified")
        pnoise_options: Mapping[str, Any] = self.specs.get('pnoise_options', {})
        measurement: Optional[Sequence[Mapping[str, Any]] = pnoise_options.pop('measurement', None)
        pnoise_dict_list = []
        for probe_info in probe_info_list:
            if len(probe_info) == 2:
                p_port, n_port = probe_info
                addl_pnoise_options = {}
            else:
                p_port, n_port, addl_pnoise_options = probe_info
            pnoise_dict_list.append(dict(
                type='PNOISE',
                param='freq',
                sweep=sweep_options,
                save_outputs=self.save_outputs,
                p_port=p_port,
                n_port=n_port,
                measurement=measurement,
                options={**pnoise_options, **addl_pnoise_options},
            ))
        sim_setup['analyses'].extend(pnoise_dict_list)

        return netlist_info_from_dict(sim_setup)
