from epidemic_v2 import *
from typing import Dict

# Модель прогнозирует развитие эпидемии в связанных с регионом домах престарелых, СИЗО

# Механика: распространение принимается односторонним - из областного/адм центра
# региона в колонию. Логит вероятности заражения хотя бы одного человека в колонии
# прямо пропорционален произведению регионального R0 на долю инфицированных
# в населении региона (да, я знаю, что это неправильно).

# После того, как заражение произошло, колония эволюционирует по своим законам как
# небольшая популяция с высоким R0 и высокими вероятностями смертности и низким числом
# аппаратов ИВЛ (предположим, что их 1 на заведение, хотя это уже слишком много).

PRISON_AGE_DISTRIBUTION = {
    '0-9':   0.000,
    '10-19': 0.052,
    '20-29': 0.181,
    '30-39': 0.319,
    '40-49': 0.298,
    '50-59': 0.130,
    '60-69': 0.019,
    '70-79': 0.001,
    '80+':   0.000
}

PRISON_EPIDEMIC_PARAMS = EpidemicParams(average_r0=7.0,
                                        hospitalization_probabilities={
                                            '0-9': 0.06,
                                            '10-19': 0.08,
                                            '20-29': 0.08,
                                            '30-39': 0.08,
                                            '40-49': 0.11,
                                            '50-59': 0.15,
                                            '60-69': 0.30,
                                            '70-79': 0.40,
                                            '80+': 0.55
                                        },
                                        icu_probabilities={
                                            '0-9':   0.10,
                                            '10-19': 0.15,
                                            '20-29': 0.15,
                                            '30-39': 0.20,
                                            '40-49': 0.25,
                                            '50-59': 0.30,
                                            '60-69': 0.40,
                                            '70-79': 0.50,
                                            '80+':   0.60
                                        },
                                        fatality_probabilities={
                                            '0-9': 0.15,
                                            '10-19': 0.20,
                                            '20-29': 0.20,
                                            '30-39': 0.25,
                                            '40-49': 0.30,
                                            '50-59': 0.35,
                                            '60-69': 0.45,
                                            '70-79': 0.55,
                                            '80+': 0.65
                                        },
                                        overflow_severity=5.0)


def compute_institutional_path(initial_population: float,
                               age_distribution: Dict[str, float],
                               params: EpidemicParams,
                               contagion_start: date):
    pop = np.array([initial_population * age_distribution[k] for k in age_distribution.keys()])
    imports = pd.Series(data=0.0, index=params.dates,dtype='float')
    imports[contagion_start] = 1.0
    sim, m_o_d, m_o, m_i_d, m_i, m_s = simulate_path_with_maxima(
        params=params,
        icu_beds=1,
        imported_cases=regional_imports[region],
        initial_population=pop)
    return SimulationResults(summary_results=sim,
                             max_overflow_date=m_o_d,
                             max_overflow=m_o,
                             max_infectious=m_i,
                             max_infectious_date=m_i_d,
                             min_susceptible=m_s,
                             r0_trajectory=params.r0_trajectory)


def main():
    sim_results_prison = compute_institutional_path(
        initial_population=3500.0,
        params=PRISON_EPIDEMIC_PARAMS,
        age_distribution=PRISON_AGE_DISTRIBUTION,
        contagion_start=date(2020, 4, 8))

    sim_results_region = simulate_region2('Мордовия')

    r0_array = sim_results_region.r0_trajectory.to_numpy(dtype=float)
    num_infectious = sim_results_region.summary_results['Infectious'].to_numpy(dtype=float)
    population = np.sum(
        [sim_results_region.summary_results['Susceptible'].to_numpy(dtype=float),
         sim_results_region.summary_results['Exposed'].to_numpy(dtype=float),
         sim_results_region.summary_results['Infectious'].to_numpy(dtype=float),
         sim_results_region.summary_results['Hospitalized'].to_numpy(dtype=float),
         sim_results_region.summary_results['Critical'].to_numpy(dtype=float),
         sim_results_region.summary_results['Overflow'].to_numpy(dtype=float),
         sim_results_region.summary_results['Recovered'].to_numpy(dtype=float)
         ],
        axis=0
    )
    frac_infectious = np.divide(num_infectious, population)
    a1: float = 1.0
    a0: float = 0.0
    zz = np.minimum(np.multiply(frac_infectious, r0_array) * a1 + a0, 1.0)
    corrected_prob_spread: np.ndarray = zz
    plt.plot(sim_results_region.simulation_dates, frac_infectious * 100.0)
    plt.show()
    plt.plot(sim_results_region.simulation_dates, corrected_prob_spread * 100.0)
    plt.show()


if __name__ == '__main__':
    main()
