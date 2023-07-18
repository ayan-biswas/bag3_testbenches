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

from typing import Any, Sequence, Optional, Mapping, Type, List, Set, Union, Tuple, Iterable

from itertools import chain
import abc
import numpy as np

from bag.design.module import Module
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimData

from pybag.core import get_cdba_name_bits

from bag3_liberty.data import parse_cdba_name

from ..schematic.generic_tb import bag3_testbenches__generic_tb


class GenericTB(TestbenchManager, abc.ABC):
    """This class provide utility methods useful for all simulations.

    Notes
    -----
    specification dictionary has the following entries in addition to the default ones:

    sim_params : Mapping[str, float]

    subclasses can define the following optional entries:

    save_outputs : Sequence[str]
        Optional.  list of nets to save in simulation data file.
    dut_pins : Sequence[str]
        list of DUT pins.
    src_list : Sequence[Mapping[str, Any]]
        list of DC sources.
    load_list : Sequence[Mapping[str, Any]]
        Optional.  List of loads.  Each dictionary has the following entries:

        pin: str
            Optional, the pin to connect to.
        nin: str
            Optional, the negative pin to connect to.
        conns: Mapping[str, str]
            Optional, the connection dictionary. Either specify pin (and nin), or specify conns.
        type : str
            the load device type.
        value : Union[float, str]
            the load parameter value.
    harnesses_list : Optional[Sequence[Mapping[str, Any]]]
        list of harnesses used in the TB with harness_idx and conns.

        - harness_idx: int
            the index of the harness cell_name from harnesses_cell
          conns: Sequence[Tuple[str, str]]
            harness connections
    pwr_domain : Mapping[str, Tuple[str, str]]
        Dictionary from individual pin names or base names to (ground, power) pin name tuple.
    sup_values : Mapping[str, Union[float, Mapping[str, float]]]
        Dictionary from supply pin name to voltage values.
    pin_values : Mapping[str, Union[int, str]]
        Dictionary from bus pin or scalar pin to the bit value as binary integer, or a pin name
        to short pins to nets.
    diff_list : Sequence[Tuple[Sequence[str], Sequence[str]]]
        Optional.  List of groups of differential pins.
    skip_src : bool
        Defaults to True.  If True, ignore multiple stimuli on same pin (only use the
        first stimuli).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._diff_lookup: Mapping[str, Tuple[Sequence[str], Sequence[str]]] = {}
        self._bit_values: Mapping[str, Union[int, str]] = {}
        super().__init__(*args, **kwargs)

    def commit(self) -> None:
        super().commit()

        specs = self.specs
        diff_list: Sequence[Tuple[Sequence[str], Sequence[str]]] = specs.get('diff_list', [])
        pin_values: Mapping[str, Union[int, str]] = specs.get('pin_values', {})

        self._diff_lookup = self.get_diff_lookup(diff_list)
        self._bit_values = self._get_pin_bit_values(pin_values)

    @property
    def save_outputs(self) -> Sequence[str]:
        save_outputs: Optional[List[str]] = self.specs.get('save_outputs', None)
        if save_outputs is None:
            return []

        out_set = set()
        for pin in save_outputs:
            pos_pins, neg_pins = self.get_diff_groups(pin)
            out_set.update(pos_pins)
            out_set.update(neg_pins)

        return list(out_set)

    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__generic_tb

    @classmethod
    def sup_var_name(cls, sup_pin: str) -> str:
        return f'v_{sup_pin}'

    @classmethod
    def sup_src_name(cls, sup_pin: str) -> str:
        return f'VDC_{sup_pin}'

    def get_bias_sources(self, sup_values: Mapping[str, Union[float, Mapping[str, float]]],
                         src_list: List[Mapping[str, Any]], src_pins: Set[str]) -> None:
        """Save bias sources and pins into src_list and src_pins.

        Side effect: will add voltage variables in self.sim_params.
        """
        sim_params = self.sim_params
        env_params = self.env_params
        for sup_pin, sup_val in sup_values.items():
            if sup_pin in src_pins:
                raise ValueError(f'Cannot add bias source on pin {sup_pin}, already used.')

            var_name = self.sup_var_name(sup_pin)
            if sup_pin == 'VSS':
                if sup_val != 0:
                    raise ValueError('VSS must be 0 volts.')
            else:
                src_list.append(dict(type='vdc', lib='analogLib', name=self.sup_src_name(sup_pin), value=var_name,
                                     conns=dict(PLUS=sup_pin, MINUS='VSS')))
                src_pins.add(sup_pin)
            if isinstance(sup_val, float) or isinstance(sup_val, int):
                sim_params[var_name] = float(sup_val)
            else:
                env_params[var_name] = dict(**sup_val)

    @classmethod
    def get_pin_supplies(cls, pin_name: str, pwr_domain: Mapping[str, Tuple[str, str]]
                         ) -> Tuple[str, str]:
        ans = pwr_domain.get(pin_name, None)
        if ans is None:
            # check if this is a r_src pin
            pin_base = cls.get_r_src_pin_base(pin_name)
            if pin_base:
                return pwr_domain[pin_base]

            # check if this is a bus pin, and pwr_domain is specified for the whole bus
            basename = parse_cdba_name(pin_name)[0]
            return pwr_domain[basename]
        return ans

    @classmethod
    def get_diff_lookup(cls, diff_list: Sequence[Tuple[Sequence[str], Sequence[str]]]
                        ) -> Mapping[str, Tuple[Sequence[str], Sequence[str]]]:
        ans = {}
        for pos_pins, neg_pins in diff_list:
            ppin_bits = [bit_name for ppin in pos_pins for bit_name in get_cdba_name_bits(ppin)]
            npin_bits = [bit_name for npin in neg_pins for bit_name in get_cdba_name_bits(npin)]
            pos_pair = (ppin_bits, npin_bits)
            neg_pair = (npin_bits, ppin_bits)
            for ppin in ppin_bits:
                ans[ppin] = pos_pair
            for npin in npin_bits:
                ans[npin] = neg_pair
        return ans

    @classmethod
    def get_r_src_pin(cls, in_pin: str) -> str:
        return in_pin + '_rs_'

    @classmethod
    def get_r_src_pin_base(cls, pin_name: str) -> str:
        return pin_name[:-4] if pin_name.endswith('_rs_') else ''

    def pre_setup(self, sch_params: Optional[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
        if sch_params is None:
            return None

        specs = self.specs
        sup_values: Mapping[str, Union[float, Mapping[str, float]]] = specs['sup_values']
        dut_pins: Sequence[str] = specs['dut_pins'] if 'dut_cell' in sch_params else []
        load_list: Sequence[Mapping[str, Any]] = specs.get('load_list', [])

        src_list: List[Mapping[str, Any]] = specs.get('src_list', [])
        src_pins = set()
        self.get_bias_sources(sup_values, src_list, src_pins)
        self.get_loads(load_list, src_list)

        dut_conns = self.get_dut_conns(dut_pins, src_pins)
        return dict(
            dut_lib=sch_params.get('dut_lib', ''),
            dut_cell=sch_params.get('dut_cell', ''),
            dut_params=sch_params.get('dut_params', None),
            dut_conns=sch_params.get('dut_conns', dut_conns),
            harnesses_cell=sch_params.get('harnesses_cell', None),
            harnesses_list=specs.get('harnesses_list', None),
            vbias_list=[],
            src_list=src_list,
        )

    def get_dut_conns(self, dut_pins: Iterable[str], src_pins: Set[str]) -> Mapping[str, str]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        ans = {}
        for pin_name in dut_pins:
            # build net expression list
            last_bit = ''
            cur_cnt = 0
            net_list = []
            for bit_name in get_cdba_name_bits(pin_name):
                if bit_name not in src_pins:
                    bit_val = self._bit_values.get(bit_name, None)
                    if bit_val is not None:
                        if isinstance(bit_val, int):
                            bit_name = self.get_pin_supplies(bit_name, pwr_domain)[bit_val]
                        else:
                            bit_name = bit_val
                if bit_name == last_bit:
                    cur_cnt += 1
                else:
                    if last_bit:
                        net_list.append(last_bit if cur_cnt == 1 else f'<*{cur_cnt}>{last_bit}')
                    last_bit = bit_name
                    cur_cnt = 1

            if last_bit:
                net_list.append(last_bit if cur_cnt == 1 else f'<*{cur_cnt}>{last_bit}')
            ans[pin_name] = ','.join(net_list)

        return ans

    def get_loads(self, load_list: Iterable[Mapping[str, Any]],
                  src_load_list: List[Mapping[str, Any]]) -> None:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        for params in load_list:
            pin = ''
            conns = {}
            pos_pins, neg_pins = [], []
            dev_type: str = params['type']
            dev_lib: str = params.get('lib', 'analogLib')
            if 'pin' in params:
                pin: str = params['pin']
                pos_pins, neg_pins = self.get_diff_groups(pin)
            elif 'conns' in params:
                conns: Mapping[str, str] = params['conns']
            elif dev_type != 'mind':
                # mind (mutual inductor) has no terminals, so "pin" or "conns" are not needed.
                raise ValueError('Specify either "pin" or "conns".')
            # some elements like ideal_balun don't need a value
            value: Optional[Union[float, str, Mapping[str, Any]]] = params.get('value', {})
            name: Optional[str] = params.get('name')
            if 'nin' in params:
                if not pin:
                    raise ValueError('If "nin" is specified, also specify "pin". Otherwise just use "conns".')
                nin: str = params['nin']
                npos_pins, nneg_pins = self.get_diff_groups(nin)

                for pin_name, nin_name in zip(chain(pos_pins, neg_pins),
                                              chain(npos_pins, nneg_pins)):
                    src_load_list.append(dict(type=dev_type, lib=dev_lib, value=value,
                                              conns=dict(PLUS=pin_name, MINUS=nin_name), name=name))
            else:
                if pin:
                    gnd_name = self.get_pin_supplies(pin, pwr_domain)[0]
                    for pin_name in chain(pos_pins, neg_pins):
                        src_load_list.append(dict(type=dev_type, lib=dev_lib, value=value,
                                                  conns=dict(PLUS=pin_name, MINUS=gnd_name), name=name))
                else:
                    src_load_list.append(dict(type=dev_type, lib=dev_lib, value=value,
                                              conns=conns, name=name))

    def get_pin_supply_values(self, pin_name: str, data: SimData) -> Tuple[np.ndarray, np.ndarray]:
        pwr_domain: Mapping[str, Tuple[str, str]] = self.specs['pwr_domain']

        gnd_pin, pwr_pin = self.get_pin_supplies(pin_name, pwr_domain)
        gnd_var = self.sup_var_name(gnd_pin)
        pwr_var = self.sup_var_name(pwr_pin)

        return self.get_param_value(gnd_var, data), self.get_param_value(pwr_var, data)

    def get_diff_groups(self, pin_name: str) -> Tuple[Sequence[str], Sequence[str]]:
        pin_base = self.get_r_src_pin_base(pin_name)
        if pin_base:
            diff_grp = self._diff_lookup.get(pin_base, None)
            if diff_grp is None:
                return [pin_name], []
            pos_pins = [self.get_r_src_pin(p_) for p_ in diff_grp[0]]
            neg_pins = [self.get_r_src_pin(p_) for p_ in diff_grp[1]]
            return pos_pins, neg_pins
        else:
            diff_grp = self._diff_lookup.get(pin_name, None)
            if diff_grp is None:
                return [pin_name], []
            return diff_grp

    def _get_pin_bit_values(self, pin_values: Mapping[str, Union[int, str]]
                            ) -> Mapping[str, Union[int, str]]:
        ans = {}
        for pin_name, pin_val in pin_values.items():
            bit_list = get_cdba_name_bits(pin_name)
            nlen = len(bit_list)
            if isinstance(pin_val, str):
                # user specify another pin to short to
                val_list = get_cdba_name_bits(pin_val)
                if len(val_list) != len(bit_list):
                    if len(val_list) == 1:
                        val_list *= nlen
                    else:
                        raise ValueError(f'Cannot connect pin {pin_name} to {pin_val}, length mismatch.')

                for bit_name, net_name in zip(bit_list, val_list):
                    pos_bits, neg_bits = self.get_diff_groups(bit_name)
                    pos_nets, neg_nets = self.get_diff_groups(net_name)
                    for p_ in pos_bits:
                        ans[p_] = pos_nets[0]
                    for p_ in neg_bits:
                        ans[p_] = neg_nets[0]
            else:
                # user specify pin values
                bin_str = bin(pin_val)[2:].zfill(nlen)
                for bit_name, val_char in zip(bit_list, bin_str):
                    pin_val = int(val_char == '1')
                    pos_bits, neg_bits = self.get_diff_groups(bit_name)
                    for p_ in pos_bits:
                        ans[p_] = pin_val
                    for p_ in neg_bits:
                        ans[p_] = pin_val ^ 1

        return ans
