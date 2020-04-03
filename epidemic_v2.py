from epidemic_params import *

import numpy as np
import pandas as pd
# from scipy import interpolate
from matplotlib import pyplot as plt
from math import ceil

from regions import regions  # as original_regions

from regions import regional_age_distribution, \
    regional_population, \
    regional_ventilators, imports_5x, imports_raw

pd.set_option('display.expand_frame_repr', False)

SIMULATION_START_DATE = date(2020, 1, 1)
DAYS_TO_SIMULATE = 366

dates = pd.date_range(SIMULATION_START_DATE, periods=DAYS_TO_SIMULATE)

regional_imports: Dict[str, pd.Series] = {}

for region in regions:
    regional_imports[region] = pd.Series(dtype='int', index=dates)
    for import_record in imports_5x[region]:
        regional_imports[region][import_record[0]] = import_record[1]


def evolve(params: EpidemicParams,
           r0: float,
           total_icu_beds: float,
           imported_cases: float,  # предполагаем пропорциональное распределение по возрастам
           old: EpidemicState) -> EpidemicState:

    new = EpidemicState()

    population_by_age = old.susceptible \
        + old.exposed + old.infectious \
        + old.hospitalized + old.critical \
        + old.overflow + old.recovered
    population = np.sum(population_by_age)

    frac_infected = np.sum(old.infectious) / population
    imported = np.round(imported_cases * (population_by_age / population))
    new_critical_d = np.zeros(shape=(NUM_AGE_GROUPS,))
    new_stabilized_d = np.zeros(shape=(NUM_AGE_GROUPS,))
    new_icu_dead_d = np.zeros(shape=(NUM_AGE_GROUPS,))
    new_overflow_stabilized_d = np.zeros(shape=(NUM_AGE_GROUPS,))
    new_overflow_dead_d = np.zeros(shape=(NUM_AGE_GROUPS,))

    for i in range(NUM_AGE_GROUPS):
        age_group = AGE_GROUPS[i]
        prev_susceptible = old.susceptible[i]
        prev_infectious = old.infectious[i]
        prev_exposed = old.exposed[i]
        prev_hospitalized = old.hospitalized[i]
        prev_critical = old.critical[i]
        prev_overflow = old.overflow[i]

        # cumulative!
        prev_intensive = old.intensive[i]
        prev_recovered = old.recovered[i]
        prev_discharged = old.discharged[i]
        prev_dead = old.dead[i]
        new_cases = ceil(float(imported[i]) + r0 * float(prev_susceptible) * frac_infected)
        new_infectious = min(prev_exposed, ceil(float(prev_exposed) / float(params.incubation_time_days)))

        # Выздоровление без госпитализации
        new_recovered = min(prev_infectious,
                            ceil(prev_infectious * params.recovery_rates[age_group]))
        new_hospitalized = min(prev_infectious - new_recovered,
                               ceil(prev_infectious * params.hospitalization_rates[age_group]))
        new_discharged = min(prev_hospitalized,
                             max(ceil(prev_hospitalized * params.discharge_rates[age_group]), 0.0))
        new_critical = min(prev_hospitalized - new_discharged,
                           ceil(prev_hospitalized * params.critical_rates[age_group]))
        new_critical_d[i] = new_critical
        new_stabilized = min(prev_critical,
                             ceil(prev_critical * params.stabilization_rates[age_group]))
        new_stabilized_d[i] = new_stabilized

        new_icu_dead = min(prev_critical - new_stabilized,
                           ceil(prev_critical * params.death_rates[age_group]))

        new_icu_dead_d[i] = new_icu_dead

        # We assume stabilization at the same rate as inside ICUs. Iffy
        # new_overflow_stabilized = min(prev_overflow,
        #                               ceil(prev_overflow * params.overflow_stabilization_rates[age_group]))
        new_overflow_stabilized = min(prev_overflow,
                                      ceil(prev_overflow * params.stabilization_rates[age_group]))
        new_overflow_stabilized_d[i] = new_overflow_stabilized

        new_overflow_dead = min(prev_overflow - new_overflow_stabilized,
                                ceil(prev_overflow * params.overflow_death_rates[age_group]))
        new_overflow_dead_d[i] = new_overflow_dead

        new.susceptible[i] = max(0.0,
                                 prev_susceptible
                                 - new_cases)
        new.exposed[i] = max(0.0,
                             prev_exposed + new_cases
                             - new_infectious)
        new.infectious[i] = max(0.0,
                                prev_infectious + new_infectious
                                - new_recovered
                                - new_hospitalized)
        new.hospitalized[i] = max(0.0,
                                  prev_hospitalized + new_hospitalized
                                  + new_stabilized
                                  + new_overflow_stabilized
                                  - new_discharged
                                  - new_critical)
        # cumulative categories
        new.recovered[i] = max(0.0,
                               prev_recovered
                               + new_recovered
                               + new_discharged)

        new.intensive[i] = prev_intensive + new_critical
        new.discharged[i] = prev_discharged + new_discharged
        new.dead[i] = prev_dead + new_icu_dead + new_overflow_dead
    # Triage and overflow
    # Move hospitalized patients according to constrained resources
    free_icu_beds = total_icu_beds - np.sum(old.critical) \
        + np.sum(new_stabilized_d) \
        + np.sum(new_icu_dead_d)
    # FIXME: triage strategy check
    # for i in reversed(range(NUM_AGE_GROUPS)):
    for i in reversed(np.argsort(old.infectious)):
        if free_icu_beds > new_critical_d[i]:
            free_icu_beds -= new_critical_d[i]
            new.critical[i] = old.critical[i] + new_critical_d[i] - new_stabilized_d[i] - new_icu_dead_d[i]
            new.overflow[i] = old.overflow[i] - new_overflow_dead_d[i] - new_overflow_stabilized_d[i]
        else:
            if free_icu_beds > 0.0:
                new_overflow = new_critical_d[i] - free_icu_beds
                new.critical[i] = old.critical[i] + free_icu_beds - new_stabilized_d[i] - new_icu_dead_d[i]
                new.overflow[i] = old.overflow[i] + new_overflow - new_overflow_stabilized_d[i] - new_overflow_dead_d[i]
                free_icu_beds = 0.0
            else:
                new.critical[i] = old.critical[i] - new_stabilized_d[i] - new_icu_dead_d[i]
                new.overflow[i] = old.overflow[i] \
                    + new_critical_d[i] \
                    - new_overflow_dead_d[i] \
                    - new_overflow_stabilized_d[i]
    # If any overflow patients are left AND there are free beds, move them back.
    # Again, move w/ lower age as priority.
    # for i in reversed(np.argsort(old.infectious)):
    for i in range(NUM_AGE_GROUPS):
        if free_icu_beds > 0.0:
            if new.overflow[i] < free_icu_beds:
                new.critical[i] += new.overflow[i]
                free_icu_beds -= new.overflow[i]
                new.overflow[i] = 0.0
            else:
                new.critical[i] += free_icu_beds
                new.overflow[i] -= free_icu_beds
                free_icu_beds = 0.0
    return new


def compute_fatalities(params: EpidemicParams,
                       icu_beds: float,
                       initial_population: np.ndarray,
                       ):
    state = EpidemicState()
    state.susceptible = initial_population.copy()

    for simdate in params.dates:
        state = evolve(
            params=params,
            r0=params.r0_trajectory[simdate],
            total_icu_beds=icu_beds,
            imported_cases=100.0,  # предполагаем пропорциональное распределение по возрастам
            old=state)

    return state.dead


def simulate_path(params: EpidemicParams,
                  icu_beds: float,
                  initial_population: np.ndarray,
                  imported_cases: pd.Series = None) -> pd.DataFrame:
    summary_results = pd.DataFrame(columns=['Susceptible',
                                            'Exposed',
                                            'Infectious',
                                            'Hospitalized',
                                            'Critical',
                                            'Overflow',
                                            'Recovered',
                                            'Dead'],
                                   index=params.dates)
    state = EpidemicState()
    state.susceptible = initial_population.copy()
    summary_results.Susceptible[params.dates[0]] = np.sum(state.susceptible)
    summary_results.Exposed[params.dates[0]] = np.sum(state.exposed)
    summary_results.Infectious[params.dates[0]] = np.sum(state.infectious)
    summary_results.Hospitalized[params.dates[0]] = np.sum(state.hospitalized)
    summary_results.Critical[params.dates[0]] = np.sum(state.critical)
    summary_results.Overflow[params.dates[0]] = np.sum(state.overflow)
    summary_results.Recovered[params.dates[0]] = np.sum(state.recovered)
    summary_results.Dead[params.dates[0]] = np.sum(state.dead)

    if imported_cases is None:
        imports = pd.Series(dtype='float',
                            index=params.dates)
        imports[params.dates[1]:params.dates[20]] = 1000.0
    else:
        imports = imported_cases

    for simdate in params.dates[1:]:

        state = evolve(
            params=params,
            r0=params.r0_trajectory[simdate],
            total_icu_beds=icu_beds,
            imported_cases=imports[simdate],  # предполагаем пропорциональное распределение по возрастам
            old=state)
        summary_results.Susceptible[simdate] = np.sum(state.susceptible)
        summary_results.Exposed[simdate] = np.sum(state.exposed)
        summary_results.Infectious[simdate] = np.sum(state.infectious)
        summary_results.Hospitalized[simdate] = np.sum(state.hospitalized)
        summary_results.Critical[simdate] = np.sum(state.critical)
        summary_results.Overflow[simdate] = np.sum(state.overflow)
        summary_results.Recovered[simdate] = np.sum(state.recovered)
        summary_results.Dead[simdate] = np.sum(state.dead)

    return summary_results


def simulate_region(region: str):
    params = EpidemicParams(mitigation_strategy=[
        (SIMULATION_START_DATE, 1.0),
        (date(2020, 1, 28), 0.95),
        (date(2020, 3, 3), 0.94),
        (date(2020, 3, 17), 0.8),
        (date(2020, 4, 1), 0.3),
        (date(2020, 4, 6), 0.1),
        (date(2020, 12, 31), 0.1)
    ], overflow_severity=2.0)
    pop = np.multiply(regional_age_distribution[region], regional_population[region])
    sim = simulate_path(params=params,
                        icu_beds=regional_ventilators[region],
                        imported_cases=regional_imports[region],
                        initial_population=pop)
    plt.scatter(params.dates, sim.Dead)
    plt.title(f'{region}: {str(int(sim.Dead[-1]))} погибших')
    plt.show()
    plt.scatter(params.dates, sim.Infectious)
    plt.title(f'{region}: число вирулентных')
    # plt.yscale('log')
    plt.show()


def main():
    simulate_region('Адыгея')
    simulate_region('Москва')
    simulate_region('Костромская область')


if __name__ == '__main__':
    main()