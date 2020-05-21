from epidemic_params import *
#
import pandas as pd
import datetime
from math import ceil#, inf
from regions import regions
from regions import regional_age_distribution, regional_population, regional_ventilators, imports_raw, passenger_flows_tutu

regional_imports: Dict[str, pd.Series] = {}
dates_I = pd.date_range(date(2020, 1, 1), end = date(2020, 12, 31), freq = 'D')
for region in regions:
    regional_imports[region] = pd.Series(dtype='int', index = dates_I)
    for import_record in imports_raw[region]:
        # Привезенные случаи учитывам для Москвы до 15.03.2020; остальные - до 01.04.2020
        if region == 'Москва':
            if import_record[0] < date(2020, 3, 15): regional_imports[region][import_record[0]] = import_record[1] * 5
        else:
            if import_record[0] < date(2020, 4, 1): regional_imports[region][import_record[0]] = import_record[1] * 5


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

    infection_rate_per_day = r0 / params.infectious_period_days
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
        # FIXME: bug spotted by DVF
        new_cases = ceil(float(imported[i]) + infection_rate_per_day * float(prev_susceptible) * frac_infected)
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

def simulate_path_with_maxima(
        params: EpidemicParams,
        icu_beds: float,
        initial_population: np.ndarray,
        imported_cases: pd.Series = None) -> Tuple[pd.DataFrame,
                                                   date,   # max overflow date
                                                   float,  # max overflow number
                                                   date,   # max infectious date
                                                   float,  # max infectious number
                                                   float,   # min susceptible
                                                  ]:
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
    max_overflow: float = 0.0
    max_overflow_date: date = params.dates[0]
    max_infectious: float = 0.0
    max_infectious_date: date = params.dates[0]
    min_susceptible: float = 0.0
    min_susceptible_date: date = params.dates[0]
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
        all_susceptible = float(np.sum(state.susceptible))
        summary_results.Susceptible[simdate] = all_susceptible
        if min_susceptible > all_susceptible:
            min_susceptible = all_susceptible
            min_susceptible_date = simdate
        summary_results.Exposed[simdate] = np.sum(state.exposed)
        all_infectious = float(np.sum(state.infectious))
        summary_results.Infectious[simdate] = all_infectious
        if max_infectious < all_infectious:
            max_infectious = all_infectious
            max_infectious_date = simdate
        summary_results.Hospitalized[simdate] = np.sum(state.hospitalized)
        summary_results.Critical[simdate] = np.sum(state.critical)
        all_overflow = float(np.sum(state.overflow))
        summary_results.Overflow[simdate] = all_overflow
        if max_overflow < all_overflow:
            max_overflow = all_overflow
            max_overflow_date = simdate
        summary_results.Recovered[simdate] = np.sum(state.recovered)
        summary_results.Dead[simdate] = np.sum(state.dead)

    return summary_results, max_overflow_date, max_overflow, max_infectious_date, max_infectious, min_susceptible
