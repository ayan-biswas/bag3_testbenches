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

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, cast
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.cache import SimulationDB, DesignInstance
from bag.simulation.measure import MeasurementManager
from bag.simulation.data import SimData
from bag.concurrent.util import GatherHelper
from bag.io.file import write_yaml

from ..sp.base import SPTB
from ..data.tran import get_first_crossings, EdgeType


class MOSSPMeas(MeasurementManager):
    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        """
        Template meas_params for yaml file is given below.
        meas_params:
            sim_envs: [tt_25, ...]
            is_nmos: True or False
            meas_type: ft_fmax or gm_ro
            vgs_sweep: {vgs_min: ..., vgs_max: ..., vgs_num: ...}   # for non Monte Carlo only; use positive values
            vgs_val: ...    # for Monte Carlo only; use positive value as is_nmos will handle the sign
            vds_val: ...    # use positive value as is_nmos will handle the sign
            tbm_specs:
                sweep_options: {...}    # use 10s of GHz to THz for ft_fmax; use KHz to MHz for gm_ro
                sim_params: {...}
                pwr_domain: {...}
                sup_values: {...}
                monte_carlo_params: {...}  # optional
        """
        helper = GatherHelper()
        sim_envs = self.specs['sim_envs']
        for sim_env in sim_envs:
            helper.append(self.async_meas_pvt(name, sim_dir / sim_env, sim_db, dut, harnesses, sim_env))

        meas_results = await helper.gather_err()
        results = {'sim_envs': np.array(sim_envs)}
        tbm_specs: Mapping[str, Any] = self.specs['tbm_specs']
        mc_params = tbm_specs.get('monte_carlo_params', {})
        for key, val in meas_results[0].items():
            results[key] = np.zeros((len(sim_envs), *val.shape), dtype=float)

        for sidx, _ in enumerate(sim_envs):
            for key, val in meas_results[sidx].items():
                results[key][sidx] = val

        meas_type: str = self.specs['meas_type']
        if mc_params:
            plot_mc_results(results, sim_dir, meas_type)
        else:
            plot_results(results, sim_dir, meas_type)
        results_yaml = {key: val.tolist() for key, val in results.items()}
        write_yaml(sim_dir / 'results.yaml', results_yaml)
        return results

    async def async_meas_pvt(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                             harnesses: Optional[Sequence[DesignInstance]], pvt: str) -> Mapping[str, Any]:
        # add port on G and D using dcblock
        load_list = [dict(pin='gport', nin='s', type='port', value={'r': 50}, name='PORTG'),
                     dict(pin='gport', nin='g', type='dcblock', value=''),
                     dict(pin='dport', nin='s', type='port', value={'r': 50}, name='PORTD'),
                     dict(pin='dport', nin='d', type='dcblock', value=''),
                     ]

        # add DC bias using dcfeed
        load_list.extend([dict(pin='gbias', nin='s', type='vdc', value='vgs'),
                          dict(pin='gbias', nin='g', type='dcfeed', value=''),
                          dict(pin='dbias', nin='s', type='vdc', value='vds'),
                          dict(pin='dbias', nin='d', type='dcfeed', value=''),
                          ])

        is_mc = 'monte_carlo_params' in self.specs['tbm_specs']
        is_nmos: bool = self.specs['is_nmos']

        tbm_specs = dict(
            **self.specs['tbm_specs'],
            load_list=load_list,
            sim_envs=[pvt],
            dut_pins=dut.pin_names,
            param_type='Y',
            ports=['PORTG', 'PORTD'],
        )

        if not is_mc:
            vgs_sweep = self.specs['vgs_sweep']
            if is_nmos:
                vgs_start, vgs_stop = vgs_sweep['vgs_min'], vgs_sweep['vgs_max']
            else:
                vgs_start, vgs_stop = -vgs_sweep['vgs_max'], -vgs_sweep['vgs_min']
            tbm_specs['swp_info'] = [('vgs', dict(type='LINEAR', start=vgs_start, stop=vgs_stop, num=vgs_sweep['vgs_num']))]

        tbm = cast(SPTB, self.make_tbm(SPTB, tbm_specs))

        # set vds
        vds_val = self.specs['vds_val']
        tbm.sim_params['vds'] = vds_val if is_nmos else -vds_val

        # set vgs
        if is_mc:
            vgs_val = self.specs['vgs_val']
            tbm.sim_params['vgs'] = vgs_val if is_nmos else -vgs_val

        sim_results = await sim_db.async_simulate_tbm_obj(name, sim_dir, dut, tbm, {'dut_conns': {'b': 's', 'd': 'd',
                                                                                                  'g': 'g', 's': 's'}},
                                                          harnesses=harnesses)
        meas_type: str = self.specs['meas_type']
        if meas_type == 'ft_fmax':
            # frequency sweep should be in 10's of GHz to THz range to reach the zero crossing point
            results = calc_ft_fmax(sim_results.data, is_mc)
        elif meas_type == 'gm_ro':
            # frequency sweep should be in KHz to MHz or single digit GHz range to avoid high frequency parasitics
            results = calc_gm_ro(sim_results.data, is_mc)
        else:
            raise NotImplementedError(f'Unrecognized meas_type = {meas_type}')
        return results


def calc_ft_fmax(sim_data: SimData, is_mc: bool) -> Mapping[str, Any]:
    freq = sim_data['freq']
    y11 = sim_data['y11']
    y12 = sim_data['y12']
    y21 = sim_data['y21']
    y22 = sim_data['y22']

    # calculate fT: freq where h21 reaches 1
    h21 = y21 / y11
    ft = get_first_crossings(freq, np.abs(h21), 1, etype=EdgeType.FALL)

    # calculate fmax: freq where unilateral gain U reaches 1
    _num = np.abs(y21 - y12) ** 2
    _den = 4 * (y11.real * y22.real - y12.real * y21.real)
    ug = _num / _den
    fmax = get_first_crossings(freq, ug, 1, etype=EdgeType.FALL)

    ans = dict(ft=np.squeeze(ft), fmax=np.squeeze(fmax))
    if not is_mc:
        ans['vgs'] = sim_data['vgs']
    return ans


def calc_gm_ro(sim_data: SimData, is_mc: bool) -> Mapping[str, Any]:
    # freq = sim_data['freq']   # for debug plotting
    y21 = sim_data['y21']
    y22 = sim_data['y22']

    gm_freq = y21.real
    ro_freq = 1 / y22.real

    gm = gm_freq.mean(axis=-1)
    ro = ro_freq.mean(axis=-1)

    ans = dict(gm=np.squeeze(gm), ro=np.squeeze(ro))
    if not is_mc:
        ans['vgs'] = sim_data['vgs']
    return ans


def plot_results(results: Mapping[str, Any], sim_dir: Path, meas_type: str) -> None:
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(6, 8))
    ax1.set(xlabel='vgs (V)')

    if meas_type == 'ft_fmax':
        ax0.set(ylabel='fT (GHz)')
        ax1.set(ylabel='fmax (GHz)')
        png_name = sim_dir / 'fT_fmax.png'
    elif meas_type == 'gm_ro':
        ax0.set(ylabel='gm (mS)')
        ax1.set(ylabel='ro (Ohm)')
        png_name = sim_dir / 'gmro.png'
    else:
        raise NotImplementedError(f'Unrecognized meas_type = {meas_type}')

    for sidx, sim_env in enumerate(results['sim_envs']):
        _vgs = results['vgs'][sidx]
        if meas_type == 'ft_fmax':
            ax0.plot(_vgs, results['ft'][sidx] * 1e-9, label=sim_env)
            ax1.plot(_vgs, results['fmax'][sidx] * 1e-9, label=sim_env)
        elif meas_type == 'gm_ro':
            ax0.plot(_vgs, results['gm'][sidx] * 1e3, label=sim_env)
            ax1.semilogy(_vgs, results['ro'][sidx], label=sim_env)

    ax0.legend()
    ax0.grid()
    ax1.legend()
    ax1.grid()
    plt.tight_layout()
    plt.savefig(png_name)
    plt.close()


def plot_mc_results(results: Mapping[str, Any], sim_dir: Path, meas_type: str) -> None:
    for sidx, sim_env in enumerate(results['sim_envs']):
        if meas_type == 'ft_fmax':
            fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(6, 8))
            _ft0 = results['ft'][sidx]
            _ft = _ft0[~np.isinf(_ft0)]
            _y0, _x0, _ = ax0.hist(_ft * 1e-9, color='skyblue', edgecolor='black')
            _mean0, _std0 = _ft.mean() * 1e-9, _ft.std() * 1e-9
            ax0.text(_mean0 + 0.5 * _std0, 0.75 * _y0.max(), f'Mean: {_mean0:.2f}\nStd: {_std0:.2f}')
            ax0.set(xlabel='fT (GHz)', title=f'MC_{len(_ft)}')

            _fmax0 = results['fmax'][sidx]
            _fmax = _fmax0[~np.isinf(_fmax0)]
            _y1, _x1, _ = ax1.hist(_fmax * 1e-9, color='skyblue', edgecolor='black')
            _mean1, _std1 = _fmax.mean() * 1e-9, _fmax.std() * 1e-9
            ax1.text(_mean1 + 0.5 * _std1, 0.75 * _y1.max(), f'Mean: {_mean1:.2f}\nStd: {_std1:.2f}')
            ax1.set(xlabel='fmax (GHz)', title=f'MC_{len(_fmax)}')
        elif meas_type == 'gm_ro':
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 8))
            _gm0 = results['gm'][sidx]
            _gm = _gm0[~np.isinf(_gm0)]
            _y0, _x0, _ = ax0.hist(_gm * 1e3, color='skyblue', edgecolor='black')
            _mean0, _std0 = _gm.mean() * 1e3, _gm.std() * 1e3
            ax0.text(_mean0 + 0.5 * _std0, 0.75 * _y0.max(), f'Mean: {_mean0:.2f}\nStd: {_std0:.2f}')
            ax0.set(xlabel='gm (mS)', title=f'MC_{len(_gm)}')

            _ro0 = results['ro'][sidx]
            _ro = _ro0[~np.isinf(_ro0)]
            _y1, _x1, _ = ax1.hist(_ro * 1e-3, color='skyblue', edgecolor='black')
            _mean1, _std1 = _ro.mean() * 1e-3, _ro.std() * 1e-3
            ax1.text(_mean1 + 0.5 * _std1, 0.75 * _y1.max(), f'Mean: {_mean1:.2f}\nStd: {_std1:.2f}')
            ax1.set(xlabel='ro (KOhm)', title=f'MC_{len(_ro)}')
        else:
            raise NotImplementedError(f'Unrecognized meas_type = {meas_type}')

        plt.tight_layout()
        plt.savefig(sim_dir / f'{sim_env}_MC.png')
        plt.close()
