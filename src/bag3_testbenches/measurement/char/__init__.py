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

from typing import Type, Optional, Any, cast, Sequence, Mapping, Tuple
from pathlib import Path
from shutil import copy
import numpy as np
import matplotlib.pyplot as plt

from bag.simulation.measure import MeasurementManager
from bag.simulation.core import TestbenchManager
from bag.simulation.data import SimNetlistInfo, netlist_info_from_dict
from bag.simulation.cache import SimulationDB, DesignInstance
from bag.design.module import Module
from bag.concurrent.util import GatherHelper
from bag.math import float_to_si_string
from bag.io.file import write_yaml

from ...schematic.char_tb_sp import bag3_testbenches__char_tb_sp


class CharSPTB(TestbenchManager):
    @classmethod
    def get_schematic_class(cls) -> Type[Module]:
        return bag3_testbenches__char_tb_sp

    def get_netlist_info(self) -> SimNetlistInfo:
        sweep_var: str = self.specs.get('sweep_var', 'freq')
        sweep_options: Mapping[str, Any] = self.specs['sweep_options']
        sp_options: Mapping[str, Any] = self.specs.get('sp_options', {})
        save_outputs: Sequence[str] = self.specs.get('save_outputs', ['plus', 'minus'])
        sp_dict = dict(type='SP',
                       param=sweep_var,
                       sweep=sweep_options,
                       param_type='Y',
                       ports=['PORTP', 'PORTM'],
                       options=sp_options,
                       save_outputs=save_outputs,
                       )

        sim_setup = self.get_netlist_info_dict()
        sim_setup['analyses'] = [sp_dict]
        return netlist_info_from_dict(sim_setup)


class CharSPMeas(MeasurementManager):
    async def async_measure_performance(self, name: str, sim_dir: Path, sim_db: SimulationDB,
                                        dut: Optional[DesignInstance],
                                        harnesses: Optional[Sequence[DesignInstance]] = None) -> Mapping[str, Any]:
        helper = GatherHelper()
        sim_envs = self.specs['sim_envs']
        ibias_list = self.specs.get('ibias_list', [0])
        for sim_env in sim_envs:
            for ibias in ibias_list:
                helper.append(self.async_meas_case(name, sim_dir / sim_env, sim_db, dut, sim_env, ibias))

        meas_results = await helper.gather_err()
        passive_type: str = self.specs['passive_type']
        results = {'sim_envs': np.array(sim_envs), 'ibias': np.array(ibias_list)}
        tbm_specs: Mapping[str, Any] = self.specs['tbm_specs']
        mc_params = tbm_specs['sp_meas'].get('monte_carlo_params', {})
        _ans0 = compute_passives(meas_results[0], passive_type)
        for key, val in _ans0.items():
            results[key] = np.zeros((len(sim_envs), len(ibias_list), *val.shape), dtype=float)

        for sidx, _ in enumerate(sim_envs):
            for iidx, _ in enumerate(ibias_list):
                midx = sidx * len(ibias_list) + iidx
                _ans = compute_passives(meas_results[midx], passive_type)
                for key, val in _ans.items():
                    results[key][sidx, iidx] = val

        if mc_params:
            plot_mc_results_hist(results, sim_dir, passive_type)
        results_yaml = {key: val.tolist() for key, val in results.items()}
        write_yaml(sim_dir / 'results.yaml', results_yaml)
        return results

    async def async_meas_case(self, name: str, sim_dir: Path, sim_db: SimulationDB, dut: Optional[DesignInstance],
                              sim_env: str, ibias: float = 0.0) -> Mapping[str, Any]:
        tbm_specs = dict(
            **self.specs['tbm_specs']['sp_meas'],
            sim_envs=[sim_env],
        )
        tbm_specs['sim_params']['idc'] = ibias
        tbm = cast(CharSPTB, self.make_tbm(CharSPTB, tbm_specs))
        tbm_name = f'{name}_{float_to_si_string(ibias)}'
        sim_dir = sim_dir / tbm_name

        passive_type: str = self.specs['passive_type']
        if passive_type == 'ind':
            ind_specs: Mapping[str, Any] = self.specs['ind_specs']
            sp_file = Path(ind_specs['sp_file'])
            ind_sp = sp_file.name
            sim_dir.mkdir(parents=True, exist_ok=True)
            copy(sp_file, sim_dir / ind_sp)
            tb_ind_specs = {'ind_sp': ind_sp, 'plus': ind_specs['plus'], 'minus': ind_specs['minus']}
            dut_plus = 'plus'
            dut_minus = 'minus'
        else:
            tb_ind_specs = None
            dut_plus = self.specs['tbm_specs']['dut_plus']
            dut_minus = self.specs['tbm_specs']['dut_minus']

        tb_params = dict(
            extracted=self.specs['tbm_specs'].get('extracted', True),
            dut_pins=dut.pin_names,
            dut_plus=dut_plus,
            dut_minus=dut_minus,
            dut_vdd=self.specs['tbm_specs'].get('dut_vdd', 'VDD'),
            dut_vss=self.specs['tbm_specs'].get('dut_vss', 'VSS'),
            passive_type=passive_type,
            ind_specs=tb_ind_specs,
        )
        sim_results = await sim_db.async_simulate_tbm_obj(tbm_name, sim_dir, dut, tbm,
                                                          tb_params=tb_params)
        data = sim_results.data
        return dict(freq=data['freq'], y11=np.squeeze(data['y11']), y12=np.squeeze(data['y12']),
                    y21=np.squeeze(data['y21']), y22=np.squeeze(data['y22']))


def estimate_cap(freq: np.ndarray, yc: np.ndarray) -> float:
    """assume yc = jwC"""
    fit = np.polyfit(2 * np.pi * freq, yc.T.imag, 1)
    return fit[0]


def estimate_ind(freq: np.ndarray, zc: np.ndarray) -> Mapping[str, float]:
    """assume res and ind in series; cap in parallel"""
    w = 2 * np.pi * freq

    # TODO: Update vector computations for Monte Carlo simulations
    # find SRF: min freq where zc.imag goes from positive to negative
    vec = (zc.imag >= 0).astype(int)
    dvec = np.diff(vec)
    dvec = np.minimum(dvec, 0)
    loc1 = dvec.nonzero()[0][0]
    # Linearly interpolate between loc and (loc + 1)
    w1, z1 = w[loc1], zc.imag[loc1]
    w2, z2 = w[loc1 + 1], zc.imag[loc1 + 1]
    w0 = w2 - z2 * (w2 - w1) / (z2 - z1)

    # upto 0.1 * SRF, assume zc is just R + jwL
    idx0 = np.where(w < 0.1 * w0)[0][-1]
    res = np.mean(zc.real[:idx0])
    _fit = np.polyfit(w[:idx0], zc.imag[:idx0], 1)
    ind = _fit[0]

    cap = 1 / (w0 * w0 * ind)
    ans = dict(ind=ind, res=res, cap=cap, srf=w0 / (2 * np.pi))

    # --- Debug plots --- #
    # plt.semilogx(freq[:idx0], zc.imag[:idx0], label='Measured')
    # plt.semilogx(freq[:idx0], w[:idx0] * ind, label='Estimated')
    # plt.xlabel('Frequency (in Hz)')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.show()

    return ans


def estimate_esd(freq: np.ndarray, yc: np.ndarray) -> Tuple[float, float]:
    """assume yc = (1/R) + jwC; returns C, R"""
    fit = np.polyfit(2 * np.pi * freq, yc.T.imag, 1)
    cap: float = fit[0]
    res: float = 1 / yc.real.mean(axis=-1)
    return cap, res


def compute_passives(meas_results: Mapping[str, Any], passive_type: str) -> Mapping[str, Any]:
    freq = meas_results['freq']

    y11 = meas_results['y11']
    y12 = meas_results['y12']
    y21 = meas_results['y21']
    y22 = meas_results['y22']

    # --- Verify yc = -y12 = -y21 is consistent --- #
    if not np.isclose(y12, y21, rtol=1e-3).all():
        plt.loglog(freq, np.abs(y12), label='y12')
        plt.loglog(freq, np.abs(y21), 'g--', label='y21')
        plt.xlabel('Frequency (in Hz)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid()
        plt.show()

    yc = - (y12 + y21) / 2
    ypp = y11 + y12
    ypm = y22 + y21

    results = dict(
        cpp=estimate_cap(freq, ypp),
        cpm=estimate_cap(freq, ypm),
    )
    if passive_type == 'cap':
        results['cc'] = estimate_cap(freq, yc)
        results['r_series'] = (1 / yc).mean(axis=-1).real
    elif passive_type == 'res':
        results['c_parallel'], results['res'] = estimate_esd(freq, yc)
        if np.mean(results['cpp']) == 0 or np.mean(results['cpm']) == 0:
            cp_est = 0
        else:
            cp_est = 1 / (1 / results['cpp'] + 1 / results['cpm'])
        if results['c_parallel'].all() > 0:
            results['bw'] = 1 / (2 * np.pi * (results['c_parallel'] + cp_est) * results['res'])

        # --- Debug plots --- #
        # warr = 2 * np.pi * freq
        # z_meas = 1 / (yc + (ypp * ypm) / (ypp + ypm))
        # z_est = 1 / (1 / results['res'] + 1j * warr * (results['c_parallel'] + cp_est))
        # plt.semilogx(freq, np.abs(z_meas), label='Measured')
        # plt.semilogx(freq, np.abs(z_est), label='Estimated')
        # plt.xlabel('Frequency (in Hz)')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid()
        # plt.show()
    elif passive_type == 'esd':
        results['cc'], results['res'] = estimate_esd(freq, yc)
    elif passive_type == 'ind':
        zc = 1 / yc
        ind_values = estimate_ind(freq, zc)
        results.update(ind_values)
        # --- Debug plots --- #
        # warr = 2 * np.pi * freq
        # z_est = 1 / (1 / (ind_values['res'] + 1j * warr * ind_values['ind']) + 1j * warr * ind_values['cap'])
        # plt.semilogx(freq, zc.imag, label='Measured')
        # plt.semilogx(freq, z_est.imag, label='Estimated')
        # plt.xlabel('Frequency (in Hz)')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.grid()
        # plt.show()
    else:
        raise ValueError(f'Unknown passive_type={passive_type}. Use "cap" or "res" or "esd" or "ind".')
    return results


def plot_mc_results_hist(results: Mapping[str, Any], sim_dir: Path, passive_type: str):
    if passive_type == 'res':
        for sidx, sim_env in enumerate(results['sim_envs']):
            _res0 = results['res'][sidx, 0]     # assume ibias is not swept
            _res = _res0[~np.isinf(_res0)]
            _y, _x, _ = plt.hist(_res, color='skyblue', edgecolor='black')
            _mean, _std = _res.mean(), _res.std()
            plt.text(_mean + 0.5 * _std, 0.75 * _y.max(), f'Mean: {_mean:.3g}\nStd: {_std:.3g}')
            plt.xlabel('res (Ohm)')
            plt.title(f'MC_{len(_res)}')
            plt.savefig(sim_dir / f'{sim_env}_MC.png')
            plt.close()
    elif passive_type == 'cap':
        for sidx, sim_env in enumerate(results['sim_envs']):
            _cap0 = results['cc'][sidx, 0]     # assume ibias is not swept
            _cap = _cap0[~np.isinf(_cap0)]
            _y, _x, _ = plt.hist(_cap, color='skyblue', edgecolor='black')
            _mean, _std = _cap.mean(), _cap.std()
            plt.text(_mean + 0.5 * _std, 0.75 * _y.max(), f'Mean: {_mean:.3g}\nStd: {_std:.3g}')
            plt.xlabel('cc (F)')
            plt.title(f'MC_{len(_cap)}')
            plt.savefig(sim_dir / f'{sim_env}_MC.png')
            plt.close()
    else:
        raise NotImplementedError(f'Monte Carlo histogram plot not yet implemented for passive_type = {passive_type}')
