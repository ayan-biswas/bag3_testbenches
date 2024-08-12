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

# -*- coding: utf-8 -*-

from typing import Mapping, Any, Optional, Sequence

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_testbenches__char_tb_sp(Module):
    """Module for library bag3_testbenches cell char_tb_sp.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'char_tb_sp.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Dict[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            extracted='True to run extracted measurements',
            dut_lib='DUT library name',
            dut_cell='DUT cell name',
            dut_pins='DUT pin names',
            dut_plus='Plus connection for DUT',
            dut_minus='Minus connection for DUT',
            dut_vdd='VDD connection for DUT, optional',
            dut_vss='VSS connection for DUT, optional',
            passive_type='"cap" or "res" or "esd" or "ind"',
            ind_specs='Optional specs for inductor',
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(dut_lib='', dut_cell='', dut_pins=None, dut_vdd='VDD', dut_vss='VSS', extracted=True,
                    ind_specs=None)

    def design(self, extracted: bool, dut_lib: str, dut_cell: str, dut_plus: str, dut_minus: str, dut_vdd: str,
               dut_vss: str, passive_type: str, ind_specs: Optional[Mapping[str, Any]],
               dut_pins: Optional[Sequence[str]]) -> None:
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        dut_conns = []
        if dut_cell:
            # handle dut pins
            dut_conns = [(_term, _term) for _term in dut_pins]

        if extracted:
            self.remove_instance('Cc')
            self.remove_instance('Cpp')
            self.remove_instance('Cpm')
            self.replace_instance_master('XDUT', dut_lib, dut_cell, keep_connections=True, static=True)
            self.reconnect_instance('XDUT', [(dut_vdd, 'VDD'), (dut_vss, 'VSS'),
                                             (dut_plus, 'plus'), (dut_minus, 'minus')])
            # TODO: handle dut_conns
        else:
            if passive_type == 'cap':
                self.remove_instance('XDUT')
            elif passive_type == 'res' or passive_type == 'esd' or passive_type == 'ind':
                self.remove_instance('Cpp')
                self.remove_instance('Cpm')
                if passive_type == 'ind':
                    self.remove_instance('XDUT')
                    ind_sp: Path = Path(ind_specs['ind_sp'])
                    _n = int(ind_sp.suffix[2:-1])
                    conns = {}
                    for idx in range(_n):
                        conns[f't{idx + 1}'] = f't{idx + 1}'
                        conns[f'b{idx + 1}'] = 'common'
                    conns[f't{ind_specs["plus"]}'] = 'plus'
                    conns[f't{ind_specs["minus"]}'] = 'minus'
                    self.design_sources_and_loads([{'conns': conns, 'type': f'n{_n}port', 'value': str(ind_sp)}], 'Cc')
                else:
                    self.remove_instance('Cc')
                    self.replace_instance_master('XDUT', dut_lib, dut_cell, keep_connections=True, static=True)
                    self.reconnect_instance('XDUT', [(dut_vdd, 'VDD'), (dut_vss, 'VSS'),
                                                     (dut_plus, 'plus'), (dut_minus, 'minus')])
            else:
                raise ValueError(f'Unknown passive_type={passive_type}. Use "cap" or "res".')

        if passive_type == 'cap' or passive_type == 'res' or passive_type == 'ind':
            self.remove_instance('IBIAS')
